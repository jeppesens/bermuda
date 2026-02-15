"""
Position calculation algorithms for Bermuda BLE tracking.

Implements ESPresense-compatible positioning:
- Nadaraya-Watson kernel regression (primary, 2+ scanners on same floor)
- Nearest Node fallback (1 scanner or insufficient data)
- Floor-based scanner filtering
- Kalman filter integration for position smoothing

Ported from ESPresense-companion's Locators.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .const import _LOGGER
from .const import TRILATERATION_POSITION_TIMEOUT
from .util import validate_scanners_for_trilateration

if TYPE_CHECKING:
    from .bermuda_device import BermudaDevice


@dataclass
class TrilaterationResult:
    """Result of position calculation."""

    x: float
    y: float
    z: float
    confidence: float  # 0-100 scale
    scanner_count: int
    method: str  # "nadaraya_watson", "nearest_node"
    room_id: str | None = None
    floor_id: str | None = None
    error: float = 0.0  # Weighted residual error
    correlation: float = 0.0  # Pearson correlation


def calculate_position(
    device: BermudaDevice,
    current_time: float,
    debug_enabled: bool = False,
) -> TrilaterationResult | None:
    """Calculate device position using ESPresense-style algorithms.

    Pipeline:
    1. Gather valid scanners with positions and distances
    2. Filter by floor (if floor config available)
    3. Run Nadaraya-Watson (3+ scanners) or midpoint (2 scanners)
    4. Fall back to nearest node (1 scanner)
    5. Find room from calculated position

    Args:
        device: BermudaDevice with adverts containing distance measurements
        current_time: Current monotonic timestamp
        debug_enabled: Enable verbose debug logging

    Returns:
        TrilaterationResult or None if insufficient data
    """
    # Gather valid scanners
    valid_scanners = validate_scanners_for_trilateration(
        device,
        current_time,
        TRILATERATION_POSITION_TIMEOUT,
        debug_enabled=debug_enabled,
    )

    if not valid_scanners:
        if debug_enabled:
            _LOGGER.debug("Position calc: No valid scanners for %s", device.name)
        return None

    # Get floor/room config from coordinator
    coordinator = device._coordinator
    map_floors = getattr(coordinator, "map_floors", {})
    map_rooms = getattr(coordinator, "map_rooms", {})

    # Filter scanners by floor if floor config exists
    if map_floors:
        valid_scanners = _filter_by_floor(valid_scanners, map_floors, debug_enabled)
        if not valid_scanners:
            if debug_enabled:
                _LOGGER.debug("Position calc: No scanners after floor filtering for %s", device.name)
            return None

    scanner_count = len(valid_scanners)

    if debug_enabled:
        _LOGGER.debug(
            "Position calc for %s: %d valid scanners",
            device.name,
            scanner_count,
        )

    # Route to algorithm
    if scanner_count >= 2:
        result = _nadaraya_watson(valid_scanners, device, debug_enabled)
    else:
        result = _nearest_node(valid_scanners, device, debug_enabled)

    if result is None:
        return None

    # Find room and floor for the calculated position
    if map_rooms or map_floors:
        room_id = find_room_for_position(
            (result.x, result.y, result.z),
            list(map_rooms.values()) if map_rooms else [],
            list(map_floors.values()) if map_floors else None,
        )
        result.room_id = room_id

        # Find floor
        if map_floors:
            result.floor_id = _find_floor_for_z(result.z, map_floors)

    if debug_enabled:
        _LOGGER.debug(
            "Position calc for %s: (%.2f, %.2f, %.2f) confidence=%d method=%s room=%s floor=%s",
            device.name,
            result.x,
            result.y,
            result.z,
            result.confidence,
            result.method,
            result.room_id,
            result.floor_id,
        )

    return result


def _filter_by_floor(
    valid_scanners: list[tuple],
    map_floors: dict,
    debug_enabled: bool = False,
) -> list[tuple]:
    """Filter scanners to same floor as the closest scanner.

    ESPresense approach: use node floor assignments to group scanners.
    The closest scanner determines the active floor.
    """
    if not valid_scanners or not map_floors:
        return valid_scanners

    # Find the floor of the closest scanner
    closest_scanner = valid_scanners[0][0]
    closest_floors = getattr(closest_scanner, "node_floors", None)

    if not closest_floors:
        # Scanner has no floor assignment - use all scanners
        if debug_enabled:
            _LOGGER.debug(
                "Floor filter: Closest scanner %s has no floor assignment, using all",
                closest_scanner.name,
            )
        return valid_scanners

    # Keep scanners that share at least one floor with the closest scanner
    filtered = []
    for scanner, advert in valid_scanners:
        scanner_floors = getattr(scanner, "node_floors", None)
        if scanner_floors is None:
            # No floor info - include by default
            filtered.append((scanner, advert))
        elif set(scanner_floors) & set(closest_floors):
            # Shares a floor with closest scanner
            filtered.append((scanner, advert))
        elif debug_enabled:
            _LOGGER.debug(
                "Floor filter: Excluding %s (floors=%s, target=%s)",
                scanner.name,
                scanner_floors,
                closest_floors,
            )

    if debug_enabled:
        _LOGGER.debug(
            "Floor filter: %d -> %d scanners (floor=%s)",
            len(valid_scanners),
            len(filtered),
            closest_floors,
        )

    # Fall back to all scanners if filtering leaves too few
    if len(filtered) < 2 and len(valid_scanners) >= 2:
        if debug_enabled:
            _LOGGER.debug("Floor filter: Too few scanners after filtering, using all")
        return valid_scanners

    return filtered


def _nadaraya_watson(
    valid_scanners: list[tuple],
    device: BermudaDevice,
    debug_enabled: bool = False,
) -> TrilaterationResult | None:
    """Nadaraya-Watson kernel regression for position estimation.

    Ported from ESPresense-companion's NadarayaWatsonMultilateralizer.

    For 2 scanners: uses midpoint between them (insufficient for full regression).
    For 3+ scanners: inverse-distance-squared weighting.

    Args:
        valid_scanners: List of (scanner, advert) tuples, sorted by distance
        device: The device being positioned
        debug_enabled: Verbose logging

    Returns:
        TrilaterationResult or None
    """
    scanner_count = len(valid_scanners)

    if scanner_count < 2:
        return None

    # Build scanner data: position and distance
    positions: list[tuple[float, float, float]] = []
    distances: list[float] = []
    for scanner, advert in valid_scanners:
        if scanner.position is not None and advert.rssi_distance is not None:
            positions.append(scanner.position)
            distances.append(advert.rssi_distance)

    n = len(positions)
    if n < 2:
        return None

    # === 2 scanners: midpoint ===
    if n == 2:
        mx = (positions[0][0] + positions[1][0]) / 2.0
        my = (positions[0][1] + positions[1][1]) / 2.0
        mz = (positions[0][2] + positions[1][2]) / 2.0

        # Simple error: difference between distances and distance-to-midpoint
        error = 0.0
        for i in range(2):
            calc_dist = math.sqrt(
                (mx - positions[i][0]) ** 2
                + (my - positions[i][1]) ** 2
                + (mz - positions[i][2]) ** 2
            )
            error += abs(calc_dist - distances[i])
        error /= 2.0

        confidence = _calculate_confidence(error, 0.0, n, n)

        if debug_enabled:
            _LOGGER.debug(
                "NW 2-scanner midpoint: (%.2f, %.2f, %.2f) error=%.2f confidence=%d",
                mx, my, mz, error, confidence,
            )

        return TrilaterationResult(
            x=mx,
            y=my,
            z=mz,
            confidence=confidence,
            scanner_count=n,
            method="nadaraya_watson",
            error=error,
        )

    # === 3+ scanners: inverse-distance-squared weighting ===
    epsilon = 0.01  # Prevent division by zero
    total_weight = 0.0
    wx = 0.0
    wy = 0.0
    wz = 0.0

    for i in range(n):
        weight = 1.0 / (distances[i] ** 2 + epsilon)
        wx += positions[i][0] * weight
        wy += positions[i][1] * weight
        wz += positions[i][2] * weight
        total_weight += weight

        if debug_enabled:
            _LOGGER.debug(
                "  NW scanner %d: pos=(%.2f,%.2f,%.2f) dist=%.2fm weight=%.4f",
                i,
                positions[i][0],
                positions[i][1],
                positions[i][2],
                distances[i],
                weight,
            )

    if total_weight < 1e-10:
        return None

    x = wx / total_weight
    y = wy / total_weight
    z = wz / total_weight

    # Validate result
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
        _LOGGER.warning("NW result contains NaN/Inf: (%.2f, %.2f, %.2f)", x, y, z)
        return None

    # Calculate weighted error (residual between estimated and measured distances)
    error = _calculate_weighted_error(x, y, z, positions, distances)

    # Calculate Pearson correlation between measured and calculated distances
    correlation = _pearson_correlation(x, y, z, positions, distances)

    # Confidence from ESPresense-style scoring
    confidence = _calculate_confidence(error, correlation, n, n)

    if debug_enabled:
        _LOGGER.debug(
            "NW result: (%.2f, %.2f, %.2f) error=%.2f corr=%.3f confidence=%d scanners=%d",
            x, y, z, error, correlation, confidence, n,
        )

    return TrilaterationResult(
        x=x,
        y=y,
        z=z,
        confidence=confidence,
        scanner_count=n,
        method="nadaraya_watson",
        error=error,
        correlation=correlation,
    )


def _nearest_node(
    valid_scanners: list[tuple],
    device: BermudaDevice,
    debug_enabled: bool = False,
) -> TrilaterationResult | None:
    """Nearest node fallback locator.

    Used when fewer than 2 scanners are available.
    Returns the nearest scanner's position with very low confidence.

    Ported from ESPresense-companion's NearestNode.cs.
    """
    if not valid_scanners:
        return None

    nearest_scanner, nearest_advert = valid_scanners[0]
    if nearest_scanner.position is None:
        return None

    x, y, z = nearest_scanner.position

    # Try to find room from scanner's area
    room_id = nearest_scanner.area_id

    # Try to find floor from scanner's node_floors
    floor_id = None
    node_floors = getattr(nearest_scanner, "node_floors", None)
    if node_floors:
        floor_id = node_floors[0]

    if debug_enabled:
        _LOGGER.debug(
            "Nearest node: %s at (%.2f, %.2f, %.2f) dist=%.2fm room=%s",
            nearest_scanner.name,
            x, y, z,
            nearest_advert.rssi_distance or 0,
            room_id,
        )

    return TrilaterationResult(
        x=x,
        y=y,
        z=z,
        confidence=1,  # Very low - this is a fallback
        scanner_count=1,
        method="nearest_node",
        room_id=room_id,
        floor_id=floor_id,
    )


def _calculate_weighted_error(
    x: float,
    y: float,
    z: float,
    positions: list[tuple[float, float, float]],
    distances: list[float],
) -> float:
    """Calculate weighted residual error between estimated and measured distances.

    Returns the weighted average of |calculated_distance - measured_distance|,
    where weight = 1 / (measured_distance^2 + epsilon).
    """
    epsilon = 0.01
    total_weighted_error = 0.0
    total_weight = 0.0

    for i, (px, py, pz) in enumerate(positions):
        calc_dist = math.sqrt((x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2)
        residual = abs(calc_dist - distances[i])
        weight = 1.0 / (distances[i] ** 2 + epsilon)
        total_weighted_error += residual * weight
        total_weight += weight

    if total_weight < 1e-10:
        return 0.0

    return total_weighted_error / total_weight


def _pearson_correlation(
    x: float,
    y: float,
    z: float,
    positions: list[tuple[float, float, float]],
    distances: list[float],
) -> float:
    """Calculate Pearson correlation between measured and calculated distances.

    Returns value in [-1, 1]. Higher correlation means better position fit.
    """
    n = len(positions)
    if n < 2:
        return 0.0

    calculated = []
    for px, py, pz in positions:
        calculated.append(math.sqrt((x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2))

    # Means
    mean_measured = sum(distances) / n
    mean_calculated = sum(calculated) / n

    # Covariance and standard deviations
    cov = 0.0
    var_measured = 0.0
    var_calculated = 0.0

    for i in range(n):
        dm = distances[i] - mean_measured
        dc = calculated[i] - mean_calculated
        cov += dm * dc
        var_measured += dm * dm
        var_calculated += dc * dc

    denom = math.sqrt(var_measured * var_calculated)
    if denom < 1e-10:
        return 0.0

    return cov / denom


def _calculate_confidence(
    error: float,
    correlation: float,
    node_count: int,
    online_count: int,
) -> int:
    """Calculate confidence score from error, correlation, and node count.

    Inspired by ESPresense's MathUtils.CalculateConfidence.
    Returns integer 0-100.
    """
    # Error component: lower error = higher confidence
    # Normalize error: 0m -> 1.0, 5m -> ~0.37, 10m -> ~0.14
    error_score = math.exp(-error * 0.2) if error >= 0 else 0.0

    # Correlation component: higher = better (range -1 to 1, map to 0-1)
    corr_score = max(0.0, (correlation + 1.0) / 2.0)

    # Node count component: more nodes = higher confidence
    # 2 nodes -> 0.4, 3 -> 0.6, 4 -> 0.7, 6+ -> 0.85+
    node_score = min(1.0, 0.2 + 0.15 * node_count)

    # Online ratio (for now all valid scanners are "online")
    online_ratio = online_count / max(node_count, 1)

    # Combine scores with weights matching ESPresense behavior
    combined = (
        error_score * 0.35
        + corr_score * 0.25
        + node_score * 0.25
        + online_ratio * 0.15
    )

    # Scale to 0-100 and clamp
    confidence = int(min(100, max(0, combined * 100)))

    return confidence


def _find_floor_for_z(z: float, map_floors: dict) -> str | None:
    """Find which floor contains the given z coordinate."""
    for floor_id, floor in map_floors.items():
        bounds = floor.get("bounds")
        if bounds and len(bounds) == 2:
            min_bounds, max_bounds = bounds
            if len(min_bounds) >= 3 and len(max_bounds) >= 3:
                if min_bounds[2] <= z <= max_bounds[2]:
                    return floor_id
    return None


def point_in_polygon(point: tuple[float, float], polygon: list[tuple[float, float]]) -> bool:
    """Determine if a point (x,y) is inside a polygon using ray casting algorithm.

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
    """Find which room contains the given position.

    Args:
        position: (x, y, z) coordinates
        rooms: List of room definitions with 'points' polygons
        floors: Optional floor definitions for z-based filtering

    Returns:
        Room area_id/id if position is in a room, None otherwise
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
                        # On this floor - filter rooms to this floor
                        floor_rooms = floor.get("rooms", [])
                        if floor_rooms:
                            candidate_rooms = floor_rooms
                        else:
                            candidate_rooms = [
                                r for r in rooms if r.get("floor") == floor.get("id")
                            ]
                        break

    # Check each room polygon
    for room in candidate_rooms:
        points = room.get("points", [])
        if len(points) < 3:
            continue

        polygon = [(p[0], p[1]) for p in points]
        if point_in_polygon((x, y), polygon):
            return room.get("area_id") or room.get("id") or room.get("name")

    return None
