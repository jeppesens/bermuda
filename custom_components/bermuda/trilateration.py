"""
Trilateration algorithms for calculating device positions.

This module provides functions to calculate the (x,y,z) position of a device
based on distances to multiple scanners with known positions.

Supports:
- 1 scanner: Distance-only (low confidence)
- 2 scanners: Circle-circle intersection (bilateration) with disambiguation
- 3 scanners: Weighted centroid or least squares
- 4+ scanners: Overdetermined weighted centroid for best accuracy
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .const import _LOGGER
from .const import CONF_MAX_VELOCITY
from .const import DEFAULT_MAX_VELOCITY
from .const import TRILATERATION_POSITION_TIMEOUT
from .util import validate_scanners_for_trilateration

if TYPE_CHECKING:
    from .bermuda_advert import BermudaAdvert
    from .bermuda_device import BermudaDevice

# Room-aware trilateration configuration
MIN_SCANNERS_FOR_ROOM_CONSTRAINT = 2  # Minimum scanners needed to apply room filtering
ROOM_BOUNDARY_MARGIN = 0.3  # Meters - allow slight overshoot of room boundaries


def _filter_colinear_scanners(
    scanner_data: list[tuple[tuple[float, float, float], float]],
    min_scanners: int = 3,
    debug_enabled: bool = False,
) -> list[tuple[tuple[float, float, float], float]]:
    """
    Filter out colinear scanners to improve trilateration geometry.

    Colinear scanners (those in a straight line) create degenerate geometry
    that produces poor trilateration results. This function selects scanners
    that are non-colinear, prioritizing the closest ones.

    Inspired by ESPresense's SelectNonColinearTransmitters approach.

    Args:
        scanner_data: List of (position, distance) tuples, should be sorted by distance
        min_scanners: Minimum number of scanners to keep
        debug_enabled: Enable verbose debug logging

    Returns:
        Filtered list with non-colinear scanners
    """
    if len(scanner_data) <= 2:
        return scanner_data  # Can't be colinear with 2 or fewer

    selected = []

    for pos, dist in scanner_data:
        if len(selected) < 2:
            # First two scanners are always added
            selected.append((pos, dist))
            continue

        # Check if this scanner is colinear with any pair of existing scanners
        is_colinear = False

        for i in range(len(selected) - 1):
            for j in range(i + 1, len(selected)):
                # Create vectors from current position to two selected positions
                v1 = (
                    selected[i][0][0] - pos[0],
                    selected[i][0][1] - pos[1],
                    selected[i][0][2] - pos[2],
                )
                v2 = (
                    selected[j][0][0] - pos[0],
                    selected[j][0][1] - pos[1],
                    selected[j][0][2] - pos[2],
                )

                # Cross product to check if vectors are parallel
                cross = (
                    v1[1] * v2[2] - v1[2] * v2[1],
                    v1[2] * v2[0] - v1[0] * v2[2],
                    v1[0] * v2[1] - v1[1] * v2[0],
                )

                # Length of cross product
                cross_length = math.sqrt(cross[0]**2 + cross[1]**2 + cross[2]**2)

                # If cross product is very small, vectors are nearly parallel (colinear)
                if cross_length < 1e-5:
                    is_colinear = True
                    if debug_enabled:
                        _LOGGER.debug(
                            "Scanner at %s is colinear with scanners %d and %d (cross_length=%.6f)",
                            pos,
                            i,
                            j,
                            cross_length,
                        )
                    break

            if is_colinear:
                break

        if not is_colinear:
            selected.append((pos, dist))
            if debug_enabled:
                _LOGGER.debug("Added non-colinear scanner at %s (total: %d)", pos, len(selected))

        # Stop if we have enough scanners
        if len(selected) >= min_scanners and len(selected) >= 4:
            break

    if debug_enabled:
        _LOGGER.debug(
            "Colinear filter: reduced from %d to %d scanners",
            len(scanner_data),
            len(selected),
        )

    return selected


def _filter_scanners_by_room(
    valid_scanners: list[tuple[BermudaDevice, BermudaAdvert]],
    map_rooms: dict | None = None,
    debug_enabled: bool = False,
) -> list[tuple[BermudaDevice, BermudaAdvert]]:
    """
    Filter scanners by room based on closest scanner's area.

    Hierarchical approach:
    1. Use closest scanner's area_id as primary room constraint
    2. Include scanners in same room
    3. If insufficient scanners (< MIN_SCANNERS_FOR_ROOM_CONSTRAINT), fall back to global

    Args:
        valid_scanners: List of (scanner, advert) tuples, sorted by distance (closest first)
        map_rooms: Optional room configuration for adjacency checks
        debug_enabled: Enable verbose debug logging

    Returns:
        Filtered list of scanners in same room as closest, or global if insufficient
    """
    if not valid_scanners:
        return valid_scanners

    if len(valid_scanners) < MIN_SCANNERS_FOR_ROOM_CONSTRAINT:
        if debug_enabled:
            _LOGGER.debug(
                "Room filter: Only %d scanner(s), using global (no filtering)",
                len(valid_scanners),
            )
        return valid_scanners

    # Get closest scanner's room
    closest_scanner, _ = valid_scanners[0]
    primary_area_id = closest_scanner.area_id

    if primary_area_id is None:
        if debug_enabled:
            _LOGGER.debug(
                "Room filter: Closest scanner '%s' has no area_id, using global",
                closest_scanner.name,
            )
        return valid_scanners

    # Filter scanners in same room
    same_room_scanners = [
        (scanner, advert)
        for scanner, advert in valid_scanners
        if scanner.area_id == primary_area_id
    ]

    if debug_enabled:
        _LOGGER.debug(
            "Room filter: Primary area=%s, found %d/%d scanners in same room",
            primary_area_id,
            len(same_room_scanners),
            len(valid_scanners),
        )

    # Fallback to global if insufficient scanners in room
    if len(same_room_scanners) < MIN_SCANNERS_FOR_ROOM_CONSTRAINT:
        if debug_enabled:
            _LOGGER.debug(
                "Room filter: Insufficient scanners in room %s (%d < %d), using global",
                primary_area_id,
                len(same_room_scanners),
                MIN_SCANNERS_FOR_ROOM_CONSTRAINT,
            )
        return valid_scanners

    return same_room_scanners


def _project_point_to_polygon(
    point: tuple[float, float],
    polygon: list[tuple[float, float]],
) -> tuple[float, float]:
    """
    Project a point to the nearest point inside or on a polygon boundary.

    If point is already inside, return unchanged.
    If outside, find closest point on polygon edge.

    Args:
        point: (x, y) coordinates to project
        polygon: List of (x, y) vertices defining polygon boundary

    Returns:
        (x, y) coordinates of projected point
    """
    if point_in_polygon(point, polygon):
        return point

    # Point is outside - find closest point on polygon edges
    px, py = point
    closest_point = point
    min_distance = float("inf")

    n = len(polygon)
    for i in range(n):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % n]

        # Find closest point on line segment p1-p2
        x1, y1 = p1
        x2, y2 = p2

        # Vector from p1 to p2
        dx = x2 - x1
        dy = y2 - y1

        # Avoid division by zero for degenerate edges
        length_squared = dx * dx + dy * dy
        if length_squared < 1e-10:
            candidate = p1
        else:
            # Parameter t for closest point on line segment
            # t = 0 -> p1, t = 1 -> p2
            t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / length_squared))
            candidate = (x1 + t * dx, y1 + t * dy)

        # Check distance to candidate point
        dist = math.sqrt((px - candidate[0])**2 + (py - candidate[1])**2)
        if dist < min_distance:
            min_distance = dist
            closest_point = candidate

    return closest_point


@dataclass
class TrilaterationResult:
    """Result of trilateration calculation."""

    x: float
    y: float
    z: float
    confidence: float  # Percentage 0-100
    scanner_count: int
    method: str  # "1-scanner", "2-scanner", "3-scanner", "4+scanner"


def calculate_position(
    device: BermudaDevice,
    current_time: float,
    debug_enabled: bool = False,
) -> TrilaterationResult | None:
    """
    Calculate device position using trilateration from scanner distances.

    Args:
        device: BermudaDevice with adverts containing distance measurements
        current_time: Current monotonic timestamp for velocity checks
        debug_enabled: Enable verbose debug logging for calculations

    Returns:
        TrilaterationResult with (x,y,z) position and confidence, or None if insufficient data
    """
    _LOGGER.debug("=== TRILATERATION START for %s ===", device.name)

    if debug_enabled:
        _LOGGER.debug("Checking %d adverts for device %s", len(device.adverts), device.name)

    # Gather valid scanner positions and distances using shared helper
    valid_scanners = validate_scanners_for_trilateration(
        device,
        current_time,
        TRILATERATION_POSITION_TIMEOUT,
        debug_enabled=debug_enabled,
    )

    if not valid_scanners:
        _LOGGER.debug("=== TRILATERATION FAILED: No valid scanners for %s ===", device.name)
        return None

    # Apply room-based scanner filtering if enabled (hierarchical approach)
    use_room_filtering = device.options.get("trilateration_use_room_filtering", True)
    if use_room_filtering:
        map_rooms = getattr(device._coordinator, "map_rooms", None)
        valid_scanners = _filter_scanners_by_room(
            valid_scanners,
            map_rooms=map_rooms,
            debug_enabled=debug_enabled,
        )

        if not valid_scanners:
            _LOGGER.debug("=== TRILATERATION FAILED: No scanners after room filtering for %s ===", device.name)
            return None

    # Extract scanner positions and distances for trilateration
    scanner_data: list[tuple[tuple[float, float, float], float]] = []
    for scanner, advert in valid_scanners:
        if scanner.position is not None and advert.rssi_distance is not None:
            scanner_data.append((scanner.position, advert.rssi_distance))
    
    # Store adverts separately for variance-based weighting (optional enhancement)
    scanner_adverts: dict[tuple[float, float, float], BermudaAdvert] = {}
    for scanner, advert in valid_scanners:
        if scanner.position is not None:
            scanner_adverts[scanner.position] = advert
    
    # Store scanner objects for room/area information
    scanner_objects: dict[tuple[float, float, float], BermudaDevice] = {}
    for scanner, advert in valid_scanners:
        if scanner.position is not None:
            scanner_objects[scanner.position] = scanner

    # Filter colinear scanners if enabled (improves geometry)
    filter_colinear = device.options.get("trilateration_filter_colinear", True)
    if filter_colinear and len(scanner_data) > 3:
        scanner_data = _filter_colinear_scanners(scanner_data, min_scanners=3, debug_enabled=debug_enabled)

    scanner_count = len(scanner_data)
    if debug_enabled:
        _LOGGER.debug("Using %d scanners for %s", scanner_count, device.name)

    # Get room configuration for boundary constraints
    map_rooms = getattr(device._coordinator, "map_rooms", None)
    primary_room = None
    if map_rooms and scanner_objects:
        # Determine primary room from closest scanner
        first_scanner_pos = scanner_data[0][0] if scanner_data else None
        if first_scanner_pos and first_scanner_pos in scanner_objects:
            first_scanner = scanner_objects[first_scanner_pos]
            if hasattr(first_scanner, "area_id") and first_scanner.area_id:
                primary_room = map_rooms.get(first_scanner.area_id)

    # Route to appropriate algorithm based on scanner count
    result = None
    if scanner_count == 1:
        if debug_enabled:
            _LOGGER.debug("Routing to 1-scanner algorithm")
        result = _calculate_position_1_scanner(device, scanner_data, current_time, debug_enabled)
    elif scanner_count == 2:
        if debug_enabled:
            _LOGGER.debug("Routing to 2-scanner bilateration algorithm")
        result = _calculate_position_2_scanners(device, scanner_data, current_time, debug_enabled)
    elif scanner_count == 3:
        if debug_enabled:
            _LOGGER.debug("Routing to 3-scanner weighted centroid algorithm")
        result = _calculate_position_3_scanners(device, scanner_data, current_time, debug_enabled, scanner_adverts, primary_room)
    else:  # 4+
        if debug_enabled:
            _LOGGER.debug("Routing to %d-scanner overdetermined algorithm", scanner_count)
        result = _calculate_position_4plus_scanners(device, scanner_data, current_time, debug_enabled, scanner_adverts, primary_room)

    if result:
        _LOGGER.debug(
            "=== TRILATERATION SUCCESS for %s: (%.2f, %.2f, %.2f) confidence=%.1f%% method=%s ===",
            device.name,
            result.x,
            result.y,
            result.z,
            result.confidence,
            result.method,
        )
    else:
        _LOGGER.warning("=== TRILATERATION FAILED: Algorithm returned None for %s ===", device.name)

    return result


def _calculate_position_1_scanner(
    device: BermudaDevice,
    scanner_data: list[tuple[tuple[float, float, float], float]],
    current_time: float,
    debug_enabled: bool = False,
) -> TrilaterationResult | None:
    """
    Single scanner - can only know distance, not direction.

    Returns scanner position with low confidence.
    """
    (sx, sy, sz), distance = scanner_data[0]

    if debug_enabled:
        _LOGGER.debug("1-scanner: Scanner at (%.2f, %.2f, %.2f), distance=%.2fm", sx, sy, sz, distance)

    # If we have previous position, maintain direction
    if device.calculated_position is not None:
        prev_x, prev_y, prev_z = device.calculated_position
        if debug_enabled:
            _LOGGER.debug("1-scanner: Previous position (%.2f, %.2f, %.2f)", prev_x, prev_y, prev_z)
        # Calculate vector from scanner to previous position
        dx = prev_x - sx
        dy = prev_y - sy
        dz = prev_z - sz
        try:
            prev_distance = math.sqrt(dx**2 + dy**2 + dz**2)
        except (ValueError, OverflowError) as e:
            _LOGGER.warning("1-scanner: Math error calculating distance for %s: %s", device.name, e)
            prev_distance = 0

        if debug_enabled:
            _LOGGER.debug("1-scanner: Previous distance from scanner: %.2fm", prev_distance)

        if prev_distance > 0.01:  # Avoid division by zero
            # Normalize and scale to current distance
            scale = distance / prev_distance
            x = sx + dx * scale
            y = sy + dy * scale
            z = sz + dz * scale

            if debug_enabled:
                _LOGGER.debug("1-scanner: Using directional method, result=(%.2f, %.2f, %.2f)", x, y, z)

            return TrilaterationResult(
                x=x,
                y=y,
                z=z,
                confidence=20.0,  # Low confidence - maintaining previous direction
                scanner_count=1,
                method="1-scanner-directional",
            )

    # No previous position - just report scanner location
    if debug_enabled:
        _LOGGER.debug("1-scanner: No previous position, using scanner location (%.2f, %.2f, %.2f)", sx, sy, sz)
    return TrilaterationResult(
        x=sx,
        y=sy,
        z=sz,
        confidence=10.0,  # Very low confidence - just at scanner
        scanner_count=1,
        method="1-scanner-position",
    )


def _calculate_position_2_scanners(
    device: BermudaDevice,
    scanner_data: list[tuple[tuple[float, float, float], float]],
    current_time: float,
    debug_enabled: bool = False,
) -> TrilaterationResult | None:
    """
    Two scanners - circle-circle intersection (bilateration).

    Returns one of two possible intersection points, using disambiguation.
    """
    (s1x, s1y, s1z), d1 = scanner_data[0]
    (s2x, s2y, s2z), d2 = scanner_data[1]

    if debug_enabled:
        _LOGGER.debug("2-scanner: Scanner1=(%.2f,%.2f,%.2f) d1=%.2fm", s1x, s1y, s1z, d1)
        _LOGGER.debug("2-scanner: Scanner2=(%.2f,%.2f,%.2f) d2=%.2fm", s2x, s2y, s2z, d2)

    # Calculate distance between scanners
    dx = s2x - s1x
    dy = s2y - s1y
    dz = s2z - s1z
    scanner_distance = math.sqrt(dx**2 + dy**2 + dz**2)

    if debug_enabled:
        _LOGGER.debug("2-scanner: Distance between scanners: %.2fm", scanner_distance)

    if scanner_distance < 0.01:
        # Scanners are at same/nearly same position - can't trilaterate
        if scanner_distance == 0.0:
            _LOGGER.warning("2-scanner: Scanners at identical position for %s", device.address)
        else:
            _LOGGER.warning("2-scanner: Scanners too close (%.3fm) for %s", scanner_distance, device.address)
        return None

    # Check if circles intersect
    if d1 + d2 < scanner_distance:
        # Circles don't reach each other
        if debug_enabled:
            _LOGGER.debug(
                "2-scanner: Circles don't intersect - d1+d2=%.2f < scanner_dist=%.2f (using fallback)",
                d1 + d2,
                scanner_distance,
            )
        # Fall back to weighted centroid
        return _weighted_centroid(scanner_data, 2, "2-scanner-fallback", device, debug_enabled)

    if abs(d1 - d2) > scanner_distance:
        # One circle entirely contains the other
        if debug_enabled:
            _LOGGER.debug(
                "2-scanner: One circle contains other - |d1-d2|=%.2f > scanner_dist=%.2f (using fallback)",
                abs(d1 - d2),
                scanner_distance,
            )
        return _weighted_centroid(scanner_data, 2, "2-scanner-fallback", device, debug_enabled)

    # Calculate intersection points using 3D sphere-sphere intersection
    # First, work in 2D plane defined by the two scanners
    # Use midpoint and perpendicular bisector approach

    # Distance from scanner1 to midpoint of intersection chord
    # Add small epsilon to prevent any floating-point edge cases
    epsilon = 1e-10
    a = (d1**2 - d2**2 + scanner_distance**2) / (2 * scanner_distance + epsilon)

    # Height of intersection point from line between scanners
    h_squared = d1**2 - a**2
    if h_squared < 0:
        # Numerical error - circles barely touch
        h_squared = 0
    h = math.sqrt(h_squared)

    if debug_enabled:
        _LOGGER.debug("2-scanner: Intersection geometry - a=%.2f, h=%.2f", a, h)

    # Midpoint along line from scanner1 to scanner2
    mid_x = s1x + (dx * a / scanner_distance)
    mid_y = s1y + (dy * a / scanner_distance)
    mid_z = s1z + (dz * a / scanner_distance)

    if debug_enabled:
        _LOGGER.debug("2-scanner: Midpoint=(%.2f, %.2f, %.2f)", mid_x, mid_y, mid_z)

    # Two intersection points perpendicular to scanner line
    # We need a perpendicular vector in 3D space
    # Use cross product with "up" vector, unless scanners are vertical
    up_vec = (0, 0, 1) if abs(dz) < scanner_distance * 0.9 else (0, 1, 0)

    # Cross product for perpendicular direction
    perp_x = dy * up_vec[2] - dz * up_vec[1]
    perp_y = dz * up_vec[0] - dx * up_vec[2]
    perp_z = dx * up_vec[1] - dy * up_vec[0]
    perp_length = math.sqrt(perp_x**2 + perp_y**2 + perp_z**2)

    if perp_length < 0.01:
        # Degenerate case - use different up vector
        up_vec = (1, 0, 0)
        perp_x = dy * up_vec[2] - dz * up_vec[1]
        perp_y = dz * up_vec[0] - dx * up_vec[2]
        perp_z = dx * up_vec[1] - dy * up_vec[0]
        perp_length = math.sqrt(perp_x**2 + perp_y**2 + perp_z**2)

    # Normalize perpendicular vector (with epsilon to prevent division by zero)
    perp_length = max(perp_length, 1e-10)
    perp_x /= perp_length
    perp_y /= perp_length
    perp_z /= perp_length

    # Two candidate positions
    pos1_x = mid_x + perp_x * h
    pos1_y = mid_y + perp_y * h
    pos1_z = mid_z + perp_z * h

    pos2_x = mid_x - perp_x * h
    pos2_y = mid_y - perp_y * h
    pos2_z = mid_z - perp_z * h

    if debug_enabled:
        _LOGGER.debug("2-scanner: Candidate 1=(%.2f, %.2f, %.2f)", pos1_x, pos1_y, pos1_z)
        _LOGGER.debug("2-scanner: Candidate 2=(%.2f, %.2f, %.2f)", pos2_x, pos2_y, pos2_z)

    # Disambiguation: choose point closer to previous position
    chosen_x, chosen_y, chosen_z = pos1_x, pos1_y, pos1_z
    confidence = 50.0  # Medium confidence for 2-scanner

    if device.calculated_position is not None and device.position_timestamp is not None:
        prev_x, prev_y, prev_z = device.calculated_position

        # Calculate distance to each candidate
        dist1 = math.sqrt((pos1_x - prev_x)**2 + (pos1_y - prev_y)**2 + (pos1_z - prev_z)**2)
        dist2 = math.sqrt((pos2_x - prev_x)**2 + (pos2_y - prev_y)**2 + (pos2_z - prev_z)**2)

        # Check velocity constraint
        time_delta = current_time - device.position_timestamp
        max_velocity = device.options.get(CONF_MAX_VELOCITY, DEFAULT_MAX_VELOCITY)
        max_movement = max_velocity * time_delta if time_delta > 0 else float("inf")

        if debug_enabled:
            _LOGGER.debug("2-scanner: Previous position=(%.2f, %.2f, %.2f)", prev_x, prev_y, prev_z)
            _LOGGER.debug("2-scanner: Distance to candidate1=%.2fm, candidate2=%.2fm", dist1, dist2)
            _LOGGER.debug(
                "2-scanner: Time delta=%.1fs, max_movement=%.2fm (%.1fm/s)",
                time_delta,
                max_movement,
                max_velocity,
            )

        # Choose closer candidate, or reject if too fast
        if dist1 <= dist2:
            if dist1 <= max_movement:
                if debug_enabled:
                    _LOGGER.info("2-scanner: Chose candidate 1 (closer, within velocity limit)")
                chosen_x, chosen_y, chosen_z = pos1_x, pos1_y, pos1_z
                confidence = 60.0  # Higher confidence with history
            else:
                # Both too far - maybe use weighted centroid?
                _LOGGER.warning("2-scanner: Both candidates exceed velocity limit (fallback)")
                return _weighted_centroid(scanner_data, 2, "2-scanner-velocity-reject", device, debug_enabled)
        else:
            if dist2 <= max_movement:
                if debug_enabled:
                    _LOGGER.info("2-scanner: Chose candidate 2 (closer, within velocity limit)")
                chosen_x, chosen_y, chosen_z = pos2_x, pos2_y, pos2_z
                confidence = 60.0
            else:
                _LOGGER.warning("2-scanner: Both candidates exceed velocity limit (fallback)")
                return _weighted_centroid(scanner_data, 2, "2-scanner-velocity-reject", device, debug_enabled)
    else:
        if debug_enabled:
            _LOGGER.debug("2-scanner: No previous position, using candidate 1")

    return TrilaterationResult(
        x=chosen_x,
        y=chosen_y,
        z=chosen_z,
        confidence=confidence,
        scanner_count=2,
        method="2-scanner-bilateration",
    )


def _calculate_position_3_scanners(
    device: BermudaDevice,
    scanner_data: list[tuple[tuple[float, float, float], float]],
    current_time: float,
    debug_enabled: bool = False,
    scanner_adverts: Mapping[tuple[float, float, float], object] | None = None,
    primary_room: dict | None = None,
) -> TrilaterationResult | None:
    """
    Three scanners - weighted centroid algorithm.

    Good for x,y positioning, potentially x,y,z if scanners at different heights.
    """
    return _weighted_centroid(scanner_data, 3, "3-scanner", device, debug_enabled, scanner_adverts, primary_room)


def _calculate_position_4plus_scanners(
    device: BermudaDevice,
    scanner_data: list[tuple[tuple[float, float, float], float]],
    current_time: float,
    debug_enabled: bool = False,
    scanner_adverts: Mapping[tuple[float, float, float], object] | None = None,
    primary_room: dict | None = None,
) -> TrilaterationResult | None:
    """
    Four or more scanners - overdetermined weighted centroid.

    Best accuracy for full 3D positioning.
    """
    return _weighted_centroid(scanner_data, len(scanner_data), "4+scanner", device, debug_enabled, scanner_adverts, primary_room)


def _weighted_centroid(
    scanner_data: list[tuple[tuple[float, float, float], float]],
    scanner_count: int,
    method: str,
    device: BermudaDevice | None = None,
    debug_enabled: bool = False,
    scanner_adverts: Mapping[tuple[float, float, float], object] | None = None,
    primary_room: dict | None = None,
) -> TrilaterationResult | None:
    """
    Calculate position using weighted centroid algorithm.

    Uses combined weighting approach inspired by ESPresense:
    - Distance weight: 1 / (distance^2 + epsilon) - closer is better
    - Index weight: (total - index) / total - prioritizes scanners in sorted order

    Since scanner_data is pre-sorted by distance (closest first), this gives
    exponentially more weight to the closest scanners.

    If primary_room is provided, calculated position will be constrained to
    room polygon boundaries.
    """
    if debug_enabled:
        _LOGGER.debug("Weighted centroid: Using %d scanners, method=%s", scanner_count, method)

    epsilon = 0.1  # Prevent division by zero

    total_weight = 0.0
    weighted_x = 0.0
    weighted_y = 0.0
    weighted_z = 0.0

    # Get configuration from device options
    use_variance_weighting = True  # default - use variance-based quality weighting
    if device:
        use_variance_weighting = device.options.get("trilateration_use_variance_weighting", True)
    
    # Disable variance weighting if scanner_adverts not provided
    if scanner_adverts is None:
        use_variance_weighting = False

    for i, ((sx, sy, sz), distance) in enumerate(scanner_data):
        # Distance-based weight: inverse square of distance
        distance_weight = 1.0 / (distance**2 + epsilon)

        # Index-based weight: closer scanners (lower index) get higher weight
        # This is inspired by ESPresense's approach, but we make it less aggressive
        # to prevent one scanner from completely dominating
        # Using square root makes the progression gentler
        index_weight = math.sqrt((scanner_count - i) / scanner_count)

        # === VARIANCE-BASED WEIGHTING (ESPresence enhancement) ===
        # Use RSSI variance to weight scanner reliability
        # Lower variance = more stable signal = higher weight
        variance_weight = 1.0
        if use_variance_weighting and scanner_adverts is not None:
            advert = scanner_adverts.get((sx, sy, sz))
            if advert is not None:
                # Use RSSI variance (in dB²) instead of distance variance
                # RSSI variance is much more stable: typically 1-20 dB² for normal BLE
                rssi_var = getattr(advert, 'rssi_variance', 0.0)
                
                # Only apply variance weighting if we have meaningful data
                # Require at least some variance to avoid division issues
                if rssi_var > 0.01:  # Ignore near-zero variance (single sample or perfect signal)
                    # Use logarithmic decay formula for robust weighting
                    # This prevents weights from collapsing to zero with noisy signals
                    # variance_weight ranges from ~1.0 (low variance) to ~0.3 (high variance)
                    # Formula: weight = 1 / (1 + log(1 + variance/threshold))
                    variance_threshold = 5.0  # RSSI variance threshold in dB²
                    variance_weight = 1.0 / (1.0 + math.log1p(rssi_var / variance_threshold))
                    
                    # Apply minimum weight floor to prevent total exclusion
                    # Even very noisy scanners contribute something
                    variance_weight = max(0.2, variance_weight)
                    
                    if debug_enabled:
                        _LOGGER.debug(
                            "  Scanner %d: rssi_variance=%.3f dB², variance_weight=%.3f",
                            i, rssi_var, variance_weight
                        )

        # Combined weight emphasizes proximity, order, and measurement quality
        weight = distance_weight * index_weight * variance_weight

        # Apply same weight to all axes
        weighted_x += sx * weight
        weighted_y += sy * weight
        weighted_z += sz * weight
        total_weight += weight

        if debug_enabled:
            _LOGGER.debug(
                "  Scanner %d: pos=(%.2f,%.2f,%.2f) dist=%.2fm "
                "dist_w=%.4f idx_w=%.2f var_w=%.3f final_w=%.4f",
                i, sx, sy, sz, distance,
                distance_weight, index_weight, variance_weight, weight,
            )

    # Safety check: ensure we have non-zero weights
    if total_weight < 1e-10:
        if debug_enabled:
            _LOGGER.warning(
                "Weighted centroid failed: total_weight=%.6f (too low, likely variance issue)",
                total_weight
            )
        # Fallback: use unweighted centroid (all scanners equal)
        total_weight = float(scanner_count)
        weighted_x = sum(pos[0] for pos, _ in scanner_data)
        weighted_y = sum(pos[1] for pos, _ in scanner_data)
        weighted_z = sum(pos[2] for pos, _ in scanner_data)
        if debug_enabled:
            _LOGGER.debug("Falling back to unweighted centroid")
    
    x = weighted_x / total_weight
    y = weighted_y / total_weight
    z = weighted_z / total_weight

    # No floor-based confidence penalty applied
    confidence_z_penalty = 1.0

    if debug_enabled:
        _LOGGER.debug(
            "Weighted centroid: Result=(%.2f, %.2f, %.2f) total_weight=%.4f",
            x,
            y,
            z,
            total_weight,
        )

    # Calculate confidence based on scanner count and variance
    # More scanners = higher confidence
    base_confidence = min(50 + (scanner_count - 2) * 15, 95)

    # Reduce confidence if distances are very different (high variance)
    distances = [d for _, d in scanner_data]
    try:
        avg_distance = sum(distances) / len(distances) if distances else 0
        variance = sum((d - avg_distance)**2 for d in distances) / len(distances) if distances else 0
        std_dev = math.sqrt(variance)
    except (ValueError, ZeroDivisionError, OverflowError) as e:
        _LOGGER.warning("Error calculating variance in weighted centroid: %s", e)
        std_dev = 0
        avg_distance = 1  # Avoid division by zero below

    # Reduce confidence if standard deviation is high relative to average
    if avg_distance > 0:
        variance_penalty = min((std_dev / avg_distance) * 20, 30)
        confidence = max(base_confidence - variance_penalty, 10)
    else:
        confidence = base_confidence

    # Validate result for NaN/Inf values
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
        _LOGGER.warning(
            "Trilateration result contains NaN/Inf values: x=%.2f, y=%.2f, z=%.2f - rejecting",
            x, y, z
        )
        return None

    # Apply room boundary constraint if primary_room is available
    if primary_room is not None:
        room_points = primary_room.get("points", [])
        if len(room_points) >= 3:  # Valid polygon
            # Check if position is outside room boundary
            original_pos = (x, y)
            if not point_in_polygon(original_pos, [(p[0], p[1]) for p in room_points]):
                # Project to nearest point on/inside polygon boundary
                projected_x, projected_y = _project_point_to_polygon(
                    original_pos,
                    [(p[0], p[1]) for p in room_points],
                )

                # Apply small margin to avoid edge cases
                if math.sqrt((projected_x - x)**2 + (projected_y - y)**2) > ROOM_BOUNDARY_MARGIN:
                    if debug_enabled:
                        _LOGGER.debug(
                            "Room constraint: position (%.2f, %.2f) outside room %s, projecting to (%.2f, %.2f)",
                            x, y, primary_room.get("area_id", "unknown"), projected_x, projected_y,
                        )
                    x, y = projected_x, projected_y
                    # Reduce confidence when position is constrained
                    confidence = max(confidence * 0.8, 10)

    # Sanity check: result should be within reasonable bounds of contributing scanners
    # Calculate deviation from result point to each scanner
    max_deviation = 0.0
    max_deviation_scanner = None
    max_measured_dist = 0.0
    max_calculated_dist = 0.0
    total_deviation = 0.0
    high_deviation_count = 0

    for pos, measured_dist in scanner_data:
        calculated_dist = math.sqrt((x - pos[0])**2 + (y - pos[1])**2 + (z - pos[2])**2)
        deviation = abs(calculated_dist - measured_dist)
        total_deviation += deviation

        if deviation > max_deviation:
            max_deviation = deviation
            max_deviation_scanner = pos
            max_measured_dist = measured_dist
            max_calculated_dist = calculated_dist

    # Adaptive tolerance: allow more error for longer distances (RSSI is less accurate far away)
    # Base tolerance of 3m + 20% of average measured distance
    base_tolerance = 3.0
    distances = [d for _, d in scanner_data]
    avg_measured_dist = sum(distances) / len(distances) if distances else 5.0
    avg_deviation = total_deviation / len(scanner_data) if scanner_data else 0.0
    tolerance = base_tolerance + (0.2 * avg_measured_dist)

    # Count scanners with high deviation
    for pos, measured_dist in scanner_data:
        calculated_dist = math.sqrt((x - pos[0])**2 + (y - pos[1])**2 + (z - pos[2])**2)
        deviation = abs(calculated_dist - measured_dist)
        if deviation > tolerance:
            high_deviation_count += 1

    # Only reject if MOST scanners show extreme deviation (>1.5x tolerance)
    # OR if average deviation is very high (>1.2x tolerance)
    # Tightened from 3.0/2.0 to improve position accuracy and reduce cross-room contamination
    reject_threshold_multiplier = 1.5
    avg_threshold_multiplier = 1.2

    should_reject = False
    reject_reason = ""

    if max_deviation > tolerance * reject_threshold_multiplier:
        if high_deviation_count >= len(scanner_data) * 0.5:  # More than half have issues
            should_reject = True
            reject_reason = f"{high_deviation_count}/{len(scanner_data)} scanners show high deviation"
        elif avg_deviation > tolerance * avg_threshold_multiplier:
            should_reject = True
            reject_reason = f"average deviation {avg_deviation:.1f}m exceeds threshold"

    if should_reject:
        _LOGGER.debug(
            "Trilateration result rejected: calculated position (%.1f, %.1f, %.1f) - %s. "
            "Max deviation %.1fm from scanner at %s (measured: %.1fm, calculated: %.1fm, tolerance: %.1fm)",
            x, y, z, reject_reason, max_deviation, max_deviation_scanner,
            max_measured_dist, max_calculated_dist, tolerance * reject_threshold_multiplier
        )
        return None

    # For moderate deviations, reduce confidence and log at debug level
    if max_deviation > tolerance:
        _LOGGER.debug(
            "Trilateration result has deviation: calculated position (%.1f, %.1f, %.1f) "
            "deviates %.1fm from scanner at %s (measured: %.1fm, calculated: %.1fm, tolerance: %.1fm) - "
            "reducing confidence (avg_dev: %.1fm, high_dev_count: %d/%d)",
            x, y, z, max_deviation, max_deviation_scanner,
            max_measured_dist, max_calculated_dist, tolerance,
            avg_deviation, high_deviation_count, len(scanner_data)
        )
        # Progressive confidence reduction based on how far over tolerance
        deviation_ratio = max_deviation / tolerance
        confidence_multiplier = max(1.0 / deviation_ratio, 0.3)  # Don't reduce below 30% of original
        confidence = max(confidence * confidence_multiplier, 10)

    # Apply z-deviation confidence penalty (from floor mismatch detection)
    confidence = max(confidence * confidence_z_penalty, 10)

    return TrilaterationResult(
        x=x,
        y=y,
        z=z,
        confidence=confidence,
        scanner_count=scanner_count,
        method=method,
    )


def point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    """
    Determine if a point (x,y) is inside a polygon using ray casting algorithm.

    Args:
        point: (x, y) coordinates to test
        polygon: List of (x, y) vertices defining polygon boundary

    Returns:
        True if point is inside polygon
    """
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def find_room_for_position(
    position: tuple[float, float, float],
    rooms: list[dict],
    floors: list[dict] | None = None,
) -> str | None:
    """
    Find which room contains the given position.

    Args:
        position: (x, y, z) coordinates
        rooms: List of room definitions with 'points' polygons
        floors: Optional floor definitions for z-based filtering

    Returns:
        Room area_id if position is in a room, None otherwise
    """
    x, y, z = position

    # Filter by floor if available
    candidate_rooms = rooms
    if floors:
        for floor in floors:
            bounds = floor.get("bounds")
            if bounds and len(bounds) == 2:
                min_bounds, max_bounds = bounds
                if len(min_bounds) >= 3 and len(max_bounds) >= 3:
                    if min_bounds[2] <= z <= max_bounds[2]:
                        # On this floor, filter rooms
                        candidate_rooms = [r for r in rooms if r.get("floor") == floor["id"]]
                        break

    # Check each room polygon
    for room in candidate_rooms:
        points = room.get("points", [])
        if len(points) < 3:  # Need at least 3 points for a polygon
            continue

        polygon = [(p[0], p[1]) for p in points]
        if point_in_polygon((x, y), polygon):
            return room.get("area_id") or room.get("id")

    return None
