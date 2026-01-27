#!/usr/bin/env python3
"""Quick test of room-aware trilateration implementation."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

# Add custom_components to path
sys.path.insert(0, "/Users/toni/Code/bermuda")

from custom_components.bermuda.trilateration import (
    _filter_scanners_by_room,
    _project_point_to_polygon,
    point_in_polygon,
)


def test_room_filtering():
    """Test room-based scanner filtering."""
    print("Testing room-based scanner filtering...")
    
    # Create mock scanners with different area_ids
    scanner1 = MagicMock()
    scanner1.name = "Living Room Scanner 1"
    scanner1.area_id = "living_room"
    scanner1.position = (0.0, 0.0, 0.0)
    
    scanner2 = MagicMock()
    scanner2.name = "Living Room Scanner 2"
    scanner2.area_id = "living_room"
    scanner2.position = (5.0, 0.0, 0.0)
    
    scanner3 = MagicMock()
    scanner3.name = "Kitchen Scanner"
    scanner3.area_id = "kitchen"
    scanner3.position = (10.0, 0.0, 0.0)
    
    advert1 = MagicMock()
    advert1.rssi_distance = 2.0
    
    advert2 = MagicMock()
    advert2.rssi_distance = 3.0
    
    advert3 = MagicMock()
    advert3.rssi_distance = 8.0
    
    # Test with all scanners (sorted by distance - closest first)
    valid_scanners = [
        (scanner1, advert1),  # Closest, in living_room
        (scanner2, advert2),  # 2nd closest, in living_room
        (scanner3, advert3),  # 3rd closest, in kitchen
    ]
    
    # Filter should keep only living room scanners
    filtered = _filter_scanners_by_room(valid_scanners, debug_enabled=True)
    
    print(f"Input: {len(valid_scanners)} scanners")
    print(f"Output: {len(filtered)} scanners")
    
    assert len(filtered) == 2, f"Expected 2 scanners, got {len(filtered)}"
    assert filtered[0][0].area_id == "living_room"
    assert filtered[1][0].area_id == "living_room"
    
    print("✓ Room filtering works correctly!\n")


def test_polygon_projection():
    """Test point projection to polygon."""
    print("Testing polygon projection...")
    
    # Square room: 0,0 to 10,10
    polygon = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    
    # Point inside - should return unchanged
    inside_point = (5.0, 5.0)
    projected = _project_point_to_polygon(inside_point, polygon)
    assert projected == inside_point, f"Inside point should not change: {projected}"
    print(f"  Point inside {inside_point} → {projected} ✓")
    
    # Point outside - should project to edge
    outside_point = (15.0, 5.0)
    projected = _project_point_to_polygon(outside_point, polygon)
    expected = (10.0, 5.0)  # Should be on right edge
    assert abs(projected[0] - expected[0]) < 0.01 and abs(projected[1] - expected[1]) < 0.01, \
        f"Expected {expected}, got {projected}"
    print(f"  Point outside {outside_point} → {projected} ✓")
    
    # Point far outside corner
    corner_point = (15.0, 15.0)
    projected = _project_point_to_polygon(corner_point, polygon)
    expected_corner = (10.0, 10.0)  # Should be at corner
    assert abs(projected[0] - expected_corner[0]) < 0.01 and abs(projected[1] - expected_corner[1]) < 0.01, \
        f"Expected {expected_corner}, got {projected}"
    print(f"  Point at corner {corner_point} → {projected} ✓")
    
    print("✓ Polygon projection works correctly!\n")


def test_point_in_polygon():
    """Test point-in-polygon basic cases."""
    print("Testing point-in-polygon...")
    
    polygon = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    
    assert point_in_polygon((5.0, 5.0), polygon) is True
    print("  Point inside square ✓")
    
    assert point_in_polygon((15.0, 5.0), polygon) is False
    print("  Point outside square ✓")
    
    assert point_in_polygon((-5.0, 5.0), polygon) is False
    print("  Point left of square ✓")
    
    print("✓ Point-in-polygon works correctly!\n")


if __name__ == "__main__":
    try:
        test_point_in_polygon()
        test_polygon_projection()
        test_room_filtering()
        print("\n=== ALL TESTS PASSED ===")
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
