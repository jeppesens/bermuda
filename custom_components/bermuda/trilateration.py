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
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .const import _LOGGER
from .const import CONF_MAX_VELOCITY
from .const import DEFAULT_MAX_VELOCITY
from .const import TRILATERATION_POSITION_TIMEOUT
from .util import validate_scanners_for_trilateration

if TYPE_CHECKING:
    from .bermuda_device import BermudaDevice


def _filter_colinear_scanners(
    scanner_data: list[tuple[tuple[float, float, float], float]],
    min_scanners: int = 3,
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
            _LOGGER.debug("Added non-colinear scanner at %s (total: %d)", pos, len(selected))

        # Stop if we have enough scanners
        if len(selected) >= min_scanners and len(selected) >= 4:
            break

    _LOGGER.debug(
        "Colinear filter: reduced from %d to %d scanners",
        len(scanner_data),
        len(selected),
    )

    return selected


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
) -> TrilaterationResult | None:
    """
    Calculate device position using trilateration from scanner distances.

    Args:
        device: BermudaDevice with adverts containing distance measurements
        current_time: Current monotonic timestamp for velocity checks

    Returns:
        TrilaterationResult with (x,y,z) position and confidence, or None if insufficient data
    """
    _LOGGER.debug("=== TRILATERATION START for %s ===", device.name)

    _LOGGER.debug("Checking %d adverts for device %s", len(device.adverts), device.name)

    # Gather valid scanner positions and distances using shared helper
    valid_scanners = validate_scanners_for_trilateration(
        device,
        current_time,
        TRILATERATION_POSITION_TIMEOUT,
    )

    if not valid_scanners:
        _LOGGER.debug("=== TRILATERATION FAILED: No valid scanners for %s ===", device.name)
        return None

    # Extract scanner positions and distances for trilateration
    scanner_data: list[tuple[tuple[float, float, float], float]] = [
        (scanner.position, advert.rssi_distance) for scanner, advert in valid_scanners
    ]

    # Filter colinear scanners if enabled (improves geometry)
    filter_colinear = device.options.get("trilateration_filter_colinear", True)
    if filter_colinear and len(scanner_data) > 3:
        scanner_data = _filter_colinear_scanners(scanner_data, min_scanners=3)

    scanner_count = len(scanner_data)
    _LOGGER.debug("Using %d scanners for %s", scanner_count, device.name)

    # Route to appropriate algorithm based on scanner count
    result = None
    if scanner_count == 1:
        _LOGGER.debug("Routing to 1-scanner algorithm")
        result = _calculate_position_1_scanner(device, scanner_data, current_time)
    elif scanner_count == 2:
        _LOGGER.debug("Routing to 2-scanner bilateration algorithm")
        result = _calculate_position_2_scanners(device, scanner_data, current_time)
    elif scanner_count == 3:
        _LOGGER.debug("Routing to 3-scanner weighted centroid algorithm")
        result = _calculate_position_3_scanners(device, scanner_data, current_time)
    else:  # 4+
        _LOGGER.debug("Routing to %d-scanner overdetermined algorithm", scanner_count)
        result = _calculate_position_4plus_scanners(device, scanner_data, current_time)

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
) -> TrilaterationResult | None:
    """
    Single scanner - can only know distance, not direction.

    Returns scanner position with low confidence.
    """
    (sx, sy, sz), distance = scanner_data[0]

    _LOGGER.debug("1-scanner: Scanner at (%.2f, %.2f, %.2f), distance=%.2fm", sx, sy, sz, distance)

    # If we have previous position, maintain direction
    if device.calculated_position is not None:
        prev_x, prev_y, prev_z = device.calculated_position
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

        _LOGGER.debug("1-scanner: Previous distance from scanner: %.2fm", prev_distance)

        if prev_distance > 0.01:  # Avoid division by zero
            # Normalize and scale to current distance
            scale = distance / prev_distance
            x = sx + dx * scale
            y = sy + dy * scale
            z = sz + dz * scale

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
) -> TrilaterationResult | None:
    """
    Two scanners - circle-circle intersection (bilateration).

    Returns one of two possible intersection points, using disambiguation.
    """
    (s1x, s1y, s1z), d1 = scanner_data[0]
    (s2x, s2y, s2z), d2 = scanner_data[1]

    _LOGGER.debug("2-scanner: Scanner1=(%.2f,%.2f,%.2f) d1=%.2fm", s1x, s1y, s1z, d1)
    _LOGGER.debug("2-scanner: Scanner2=(%.2f,%.2f,%.2f) d2=%.2fm", s2x, s2y, s2z, d2)

    # Calculate distance between scanners
    dx = s2x - s1x
    dy = s2y - s1y
    dz = s2z - s1z
    scanner_distance = math.sqrt(dx**2 + dy**2 + dz**2)

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
        _LOGGER.debug(
            "2-scanner: Circles don't intersect - d1+d2=%.2f < scanner_dist=%.2f (using fallback)",
            d1 + d2,
            scanner_distance,
        )
        # Fall back to weighted centroid
        return _weighted_centroid(scanner_data, 2, "2-scanner-fallback", device)

    if abs(d1 - d2) > scanner_distance:
        # One circle entirely contains the other
        _LOGGER.debug(
            "2-scanner: One circle contains other - |d1-d2|=%.2f > scanner_dist=%.2f (using fallback)",
            abs(d1 - d2),
            scanner_distance,
        )
        return _weighted_centroid(scanner_data, 2, "2-scanner-fallback", device)

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

    _LOGGER.debug("2-scanner: Intersection geometry - a=%.2f, h=%.2f", a, h)

    # Midpoint along line from scanner1 to scanner2
    mid_x = s1x + (dx * a / scanner_distance)
    mid_y = s1y + (dy * a / scanner_distance)
    mid_z = s1z + (dz * a / scanner_distance)

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
                _LOGGER.info("2-scanner: Chose candidate 1 (closer, within velocity limit)")
                chosen_x, chosen_y, chosen_z = pos1_x, pos1_y, pos1_z
                confidence = 60.0  # Higher confidence with history
            else:
                # Both too far - maybe use weighted centroid?
                _LOGGER.warning("2-scanner: Both candidates exceed velocity limit (fallback)")
                return _weighted_centroid(scanner_data, 2, "2-scanner-velocity-reject", device)
        else:
            if dist2 <= max_movement:
                _LOGGER.info("2-scanner: Chose candidate 2 (closer, within velocity limit)")
                chosen_x, chosen_y, chosen_z = pos2_x, pos2_y, pos2_z
                confidence = 60.0
            else:
                _LOGGER.warning("2-scanner: Both candidates exceed velocity limit (fallback)")
                return _weighted_centroid(scanner_data, 2, "2-scanner-velocity-reject", device)
    else:
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
) -> TrilaterationResult | None:
    """
    Three scanners - weighted centroid algorithm.

    Good for x,y positioning, potentially x,y,z if scanners at different heights.
    """
    return _weighted_centroid(scanner_data, 3, "3-scanner", device)


def _calculate_position_4plus_scanners(
    device: BermudaDevice,
    scanner_data: list[tuple[tuple[float, float, float], float]],
    current_time: float,
) -> TrilaterationResult | None:
    """
    Four or more scanners - overdetermined weighted centroid.

    Best accuracy for full 3D positioning.
    """
    return _weighted_centroid(scanner_data, len(scanner_data), "4+scanner", device)


def _weighted_centroid(
    scanner_data: list[tuple[tuple[float, float, float], float]],
    scanner_count: int,
    method: str,
    device: BermudaDevice | None = None,
) -> TrilaterationResult | None:
    """
    Calculate position using weighted centroid algorithm.

    Uses combined weighting approach inspired by ESPresense:
    - Distance weight: 1 / (distance^2 + epsilon) - closer is better
    - Index weight: (total - index) / total - prioritizes scanners in sorted order
    - Floor attenuation: Heavily penalizes scanners on different floors (z-difference > 1.5m)
    - Wall attenuation: Gently penalizes horizontal distance (simulates walls every ~5m)

    Since scanner_data is pre-sorted by distance (closest first), this gives
    exponentially more weight to the closest scanners.
    """
    _LOGGER.debug("Weighted centroid: Using %d scanners, method=%s", scanner_count, method)

    epsilon = 0.1  # Prevent division by zero

    total_weight = 0.0
    total_z_weight = 0.0  # Separate weight for z-axis
    weighted_x = 0.0
    weighted_y = 0.0
    weighted_z = 0.0
    
    # Get attenuation factors from device options
    floor_attenuation = 0.3  # default - reduce to 30% for different floors
    wall_attenuation = 0.7   # default - reduce by 30% per 5m segment
    if device:
        floor_attenuation = device.options.get("trilateration_floor_attenuation", 0.3)
        wall_attenuation = device.options.get("trilateration_wall_attenuation", 0.7)
    
    # Get reference position from closest scanner for floor/wall detection
    closest_x, closest_y, closest_z = scanner_data[0][0] if scanner_data else (0, 0, 0)

    for i, ((sx, sy, sz), distance) in enumerate(scanner_data):
        # Distance-based weight: inverse square of distance
        distance_weight = 1.0 / (distance**2 + epsilon)

        # Index-based weight: closer scanners (lower index) get higher weight
        # This is inspired by ESPresense's approach, but we make it less aggressive
        # to prevent one scanner from completely dominating
        # Using square root makes the progression gentler
        index_weight = math.sqrt((scanner_count - i) / scanner_count)

        # Combined weight emphasizes both proximity and order
        weight = distance_weight * index_weight
        
        # === FLOOR ATTENUATION ===
        # Calculate z-difference from closest scanner
        z_diff = abs(sz - closest_z)
        
        # If scanner is on a different floor (>1.5m z-difference), heavily penalize
        # Floors attenuate signals much more than walls
        if z_diff > 1.5:
            floor_penalty = floor_attenuation  # e.g., 0.3 = reduce to 30% weight
            weight *= floor_penalty
            _LOGGER.debug(
                "  Scanner %d: Different floor detected (z_diff=%.1fm), applying floor penalty %.2f",
                i, z_diff, floor_penalty
            )
        
        # === WALL ATTENUATION ===
        # Calculate horizontal distance from closest scanner (ignoring z)
        horizontal_dist = math.sqrt((sx - closest_x)**2 + (sy - closest_y)**2)
        
        # For every ~5m horizontal distance, reduce weight (simulates walls)
        # This is gentler than floor attenuation since walls are thinner
        if horizontal_dist > 5.0:
            wall_segments = int(horizontal_dist / 5.0)
            wall_penalty = wall_attenuation ** wall_segments  # e.g., 0.7^2 for 10m
            weight *= wall_penalty
            _LOGGER.debug(
                "  Scanner %d: Horizontal distance %.1fm (%d wall segments), applying wall penalty %.2f",
                i, horizontal_dist, wall_segments, wall_penalty
            )
        
        # Z-axis specific weight: heavily favor closest scanner's height
        # This prevents floor mixing when device is upstairs but downstairs scanners also hear it
        z_weight = weight * (1.0 / (i + 1))  # First scanner gets full weight, others decay rapidly

        weighted_x += sx * weight
        weighted_y += sy * weight
        weighted_z += sz * z_weight  # Use z-specific weight
        total_weight += weight
        total_z_weight += z_weight

        _LOGGER.debug(
            "  Scanner %d: pos=(%.2f,%.2f,%.2f) dist=%.2fm z_diff=%.1fm horiz_dist=%.1fm "
            "dist_w=%.4f idx_w=%.2f final_w=%.4f z_w=%.4f",
            i, sx, sy, sz, distance, z_diff, horizontal_dist,
            distance_weight, index_weight, weight, z_weight,
        )

    x = weighted_x / total_weight
    y = weighted_y / total_weight
    z = weighted_z / total_z_weight  # Use z-specific total weight

    _LOGGER.debug(
        "Weighted centroid: Result=(%.2f, %.2f, %.2f) total_weight=%.4f total_z_weight=%.4f",
        x,
        y,
        z,
        total_weight,
        total_z_weight,
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

    # Only reject if MOST scanners show extreme deviation (>3x tolerance)
    # OR if average deviation is very high (>2x tolerance)
    # This prevents rejecting due to one bad scanner while allowing weighted centroid approximations
    reject_threshold_multiplier = 3.0
    avg_threshold_multiplier = 2.0

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
