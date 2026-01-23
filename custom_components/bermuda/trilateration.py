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

from .const import (
    _LOGGER,
    CONF_MAX_VELOCITY,
    DEFAULT_MAX_VELOCITY,
    TRILATERATION_POSITION_TIMEOUT,
)

if TYPE_CHECKING:
    from .bermuda_device import BermudaDevice


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
    # Gather valid scanner positions and distances
    scanner_data: list[tuple[tuple[float, float, float], float]] = []

    for advert in device.adverts.values():
        # Get the scanner device
        scanner = device._coordinator.devices.get(advert.scanner_address)
        if scanner is None or not scanner.is_scanner:
            continue

        # Scanner must have position configured
        if scanner.position is None:
            continue

        # Distance must be valid and recent
        if advert.rssi_distance is None or advert.rssi_distance <= 0:
            continue

        # Advert must be recent (not stale)
        if current_time - advert.stamp > TRILATERATION_POSITION_TIMEOUT:
            continue

        scanner_data.append((scanner.position, advert.rssi_distance))

    if not scanner_data:
        return None

    scanner_count = len(scanner_data)

    # Route to appropriate algorithm based on scanner count
    if scanner_count == 1:
        return _calculate_position_1_scanner(device, scanner_data, current_time)
    elif scanner_count == 2:
        return _calculate_position_2_scanners(device, scanner_data, current_time)
    elif scanner_count == 3:
        return _calculate_position_3_scanners(device, scanner_data, current_time)
    else:  # 4+
        return _calculate_position_4plus_scanners(device, scanner_data, current_time)


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

    # If we have previous position, maintain direction
    if device.calculated_position is not None:
        prev_x, prev_y, prev_z = device.calculated_position
        # Calculate vector from scanner to previous position
        dx = prev_x - sx
        dy = prev_y - sy
        dz = prev_z - sz
        prev_distance = math.sqrt(dx**2 + dy**2 + dz**2)

        if prev_distance > 0.01:  # Avoid division by zero
            # Normalize and scale to current distance
            scale = distance / prev_distance
            x = sx + dx * scale
            y = sy + dy * scale
            z = sz + dz * scale

            return TrilaterationResult(
                x=x,
                y=y,
                z=z,
                confidence=20.0,  # Low confidence - maintaining previous direction
                scanner_count=1,
                method="1-scanner-directional",
            )

    # No previous position - just report scanner location
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

    # Calculate distance between scanners
    dx = s2x - s1x
    dy = s2y - s1y
    dz = s2z - s1z
    scanner_distance = math.sqrt(dx**2 + dy**2 + dz**2)

    if scanner_distance < 0.01:
        # Scanners are at same position - can't trilaterate
        _LOGGER.debug("Scanners too close for 2-scanner trilateration: %s", device.address)
        return None

    # Check if circles intersect
    if d1 + d2 < scanner_distance:
        # Circles don't reach each other
        _LOGGER.debug(
            "Circles don't intersect for device %s: d1=%0.2f, d2=%0.2f, scanner_dist=%0.2f",
            device.address,
            d1,
            d2,
            scanner_distance,
        )
        # Fall back to weighted centroid
        return _weighted_centroid(scanner_data, 2, "2-scanner-fallback")

    if abs(d1 - d2) > scanner_distance:
        # One circle entirely contains the other
        _LOGGER.debug("One circle contains the other for device %s", device.address)
        return _weighted_centroid(scanner_data, 2, "2-scanner-fallback")

    # Calculate intersection points using 3D sphere-sphere intersection
    # First, work in 2D plane defined by the two scanners
    # Use midpoint and perpendicular bisector approach

    # Distance from scanner1 to midpoint of intersection chord
    a = (d1**2 - d2**2 + scanner_distance**2) / (2 * scanner_distance)

    # Height of intersection point from line between scanners
    h_squared = d1**2 - a**2
    if h_squared < 0:
        # Numerical error - circles barely touch
        h_squared = 0
    h = math.sqrt(h_squared)

    # Midpoint along line from scanner1 to scanner2
    mid_x = s1x + (dx * a / scanner_distance)
    mid_y = s1y + (dy * a / scanner_distance)
    mid_z = s1z + (dz * a / scanner_distance)

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

    # Normalize perpendicular vector
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

        # Choose closer candidate, or reject if too fast
        if dist1 <= dist2:
            if dist1 <= max_movement:
                chosen_x, chosen_y, chosen_z = pos1_x, pos1_y, pos1_z
                confidence = 60.0  # Higher confidence with history
            else:
                # Both too far - maybe use weighted centroid?
                return _weighted_centroid(scanner_data, 2, "2-scanner-velocity-reject")
        else:
            if dist2 <= max_movement:
                chosen_x, chosen_y, chosen_z = pos2_x, pos2_y, pos2_z
                confidence = 60.0
            else:
                return _weighted_centroid(scanner_data, 2, "2-scanner-velocity-reject")

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
    return _weighted_centroid(scanner_data, 3, "3-scanner")


def _calculate_position_4plus_scanners(
    device: BermudaDevice,
    scanner_data: list[tuple[tuple[float, float, float], float]],
    current_time: float,
) -> TrilaterationResult | None:
    """
    Four or more scanners - overdetermined weighted centroid.

    Best accuracy for full 3D positioning.
    """
    return _weighted_centroid(scanner_data, len(scanner_data), "4+scanner")


def _weighted_centroid(
    scanner_data: list[tuple[tuple[float, float, float], float]],
    scanner_count: int,
    method: str,
) -> TrilaterationResult:
    """
    Calculate position using weighted centroid algorithm.

    Weight = 1 / (distance^2 + epsilon)
    
    This gives higher weight to closer scanners, which typically have
    more accurate distance measurements.
    """
    epsilon = 0.1  # Prevent division by zero

    total_weight = 0.0
    weighted_x = 0.0
    weighted_y = 0.0
    weighted_z = 0.0

    for (sx, sy, sz), distance in scanner_data:
        weight = 1.0 / (distance**2 + epsilon)
        weighted_x += sx * weight
        weighted_y += sy * weight
        weighted_z += sz * weight
        total_weight += weight

    x = weighted_x / total_weight
    y = weighted_y / total_weight
    z = weighted_z / total_weight

    # Calculate confidence based on scanner count and variance
    # More scanners = higher confidence
    base_confidence = min(50 + (scanner_count - 2) * 15, 95)
    
    # Reduce confidence if distances are very different (high variance)
    distances = [d for _, d in scanner_data]
    avg_distance = sum(distances) / len(distances)
    variance = sum((d - avg_distance)**2 for d in distances) / len(distances)
    std_dev = math.sqrt(variance)
    
    # Reduce confidence if standard deviation is high relative to average
    if avg_distance > 0:
        variance_penalty = min((std_dev / avg_distance) * 20, 30)
        confidence = max(base_confidence - variance_penalty, 10)
    else:
        confidence = base_confidence

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
