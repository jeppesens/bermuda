"""Tests for trilateration algorithms."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from custom_components.bermuda.trilateration import TrilaterationResult
from custom_components.bermuda.trilateration import calculate_position
from custom_components.bermuda.trilateration import find_room_for_position
from custom_components.bermuda.trilateration import point_in_polygon


@pytest.fixture
def mock_device():
    """Create a mock BermudaDevice for testing."""
    device = MagicMock()
    device.name = "Test Device"
    device.address = "aa:bb:cc:dd:ee:ff"
    device.adverts = {}
    device.calculated_position = None
    device.position_timestamp = None
    device.options = {"max_velocity": 3.0}
    device._coordinator = MagicMock()
    device._coordinator.devices = {}
    return device


@pytest.fixture
def mock_scanner():
    """Create a mock scanner device."""
    def _create_scanner(address, position, name="Scanner"):
        scanner = MagicMock()
        scanner.address = address
        scanner.name = name
        scanner.position = position
        scanner.is_scanner = True
        return scanner
    return _create_scanner


@pytest.fixture
def mock_advert():
    """Create a mock advertisement."""
    def _create_advert(scanner_address, rssi_distance, stamp):
        advert = MagicMock()
        advert.scanner_address = scanner_address
        advert.rssi_distance = rssi_distance
        advert.stamp = stamp
        return advert
    return _create_advert


class TestPointInPolygon:
    """Test point-in-polygon algorithm."""

    def test_point_inside_square(self):
        """Test point clearly inside a square."""
        polygon = [(0, 0), (10, 0), (10, 10), (0, 10)]
        assert point_in_polygon((5, 5), polygon) is True

    def test_point_outside_square(self):
        """Test point clearly outside a square."""
        polygon = [(0, 0), (10, 0), (10, 10), (0, 10)]
        assert point_in_polygon((15, 15), polygon) is False

    def test_point_on_edge(self):
        """Test point on polygon edge."""
        polygon = [(0, 0), (10, 0), (10, 10), (0, 10)]
        # Point on edge - behavior may vary by implementation
        assert point_in_polygon((5, 0), polygon) in [True, False]

    def test_point_at_vertex(self):
        """Test point at polygon vertex."""
        polygon = [(0, 0), (10, 0), (10, 10), (0, 10)]
        assert point_in_polygon((0, 0), polygon) in [True, False]

    def test_complex_polygon(self):
        """Test with irregular polygon shape."""
        polygon = [(0, 0), (5, 0), (5, 5), (10, 5), (10, 10), (0, 10)]
        assert point_in_polygon((2, 2), polygon) is True
        assert point_in_polygon((7, 2), polygon) is False
        assert point_in_polygon((7, 7), polygon) is True

    def test_triangle(self):
        """Test with minimal polygon (triangle)."""
        polygon = [(0, 0), (10, 0), (5, 10)]
        assert point_in_polygon((5, 5), polygon) is True
        assert point_in_polygon((1, 9), polygon) is False


class TestFindRoomForPosition:
    """Test room detection from position."""

    def test_position_in_single_room(self):
        """Test finding room when position is clearly inside."""
        rooms = [
            {
                "id": "room1",
                "area_id": "living_room",
                "name": "Living Room",
                "points": [[0, 0], [10, 0], [10, 10], [0, 10]],
            }
        ]
        result = find_room_for_position((5, 5, 1), rooms)
        assert result == "living_room"

    def test_position_outside_all_rooms(self):
        """Test when position is not in any room."""
        rooms = [
            {
                "id": "room1",
                "area_id": "living_room",
                "name": "Living Room",
                "points": [[0, 0], [10, 0], [10, 10], [0, 10]],
            }
        ]
        result = find_room_for_position((15, 15, 1), rooms)
        assert result is None

    def test_multiple_rooms(self):
        """Test with multiple rooms."""
        rooms = [
            {
                "id": "room1",
                "area_id": "living_room",
                "points": [[0, 0], [10, 0], [10, 10], [0, 10]],
            },
            {
                "id": "room2",
                "area_id": "kitchen",
                "points": [[10, 0], [20, 0], [20, 10], [10, 10]],
            },
        ]
        assert find_room_for_position((5, 5, 1), rooms) == "living_room"
        assert find_room_for_position((15, 5, 1), rooms) == "kitchen"

    def test_floor_filtering(self):
        """Test room detection with floor filtering."""
        floors = [
            {"id": "ground", "bounds": [[0, 0, 0], [20, 20, 3]]},
            {"id": "first", "bounds": [[0, 0, 3], [20, 20, 6]]},
        ]
        rooms = [
            {
                "id": "ground_room",
                "area_id": "ground_living",
                "floor": "ground",
                "points": [[0, 0], [10, 0], [10, 10], [0, 10]],
            },
            {
                "id": "first_room",
                "area_id": "first_bedroom",
                "floor": "first",
                "points": [[0, 0], [10, 0], [10, 10], [0, 10]],
            },
        ]
        # Same x,y but different z should map to different floors
        assert find_room_for_position((5, 5, 1), rooms, floors) == "ground_living"
        assert find_room_for_position((5, 5, 4), rooms, floors) == "first_bedroom"

    def test_invalid_room_polygon(self):
        """Test handling of invalid room data."""
        rooms = [
            {"id": "invalid", "area_id": "invalid_room", "points": [[0, 0]]},  # Only 1 point
        ]
        result = find_room_for_position((5, 5, 1), rooms)
        assert result is None

    def test_fallback_to_id_when_no_area_id(self):
        """Test using room id when area_id missing."""
        rooms = [
            {
                "id": "room1",
                "name": "Room 1",
                "points": [[0, 0], [10, 0], [10, 10], [0, 10]],
            }
        ]
        result = find_room_for_position((5, 5, 1), rooms)
        assert result == "room1"


class TestCalculatePosition1Scanner:
    """Test 1-scanner position calculation."""

    def test_1_scanner_no_previous_position(self, mock_device, mock_scanner, mock_advert):
        """Test 1 scanner with no history - should return scanner position."""
        scanner = mock_scanner("scanner1", (10.0, 10.0, 1.0), "Scanner1")
        mock_device._coordinator.devices = {"scanner1": scanner}
        mock_device.adverts = {"scanner1": mock_advert("scanner1", 5.0, 100.0)}

        result = calculate_position(mock_device, 101.0)

        assert result is not None
        assert result.scanner_count == 1
        assert result.method in ["1-scanner-position", "1-scanner-directional"]
        assert result.confidence <= 20.0
        # Should be at or near scanner position
        assert abs(result.x - 10.0) < 0.1
        assert abs(result.y - 10.0) < 0.1

    def test_1_scanner_with_previous_position(self, mock_device, mock_scanner, mock_advert):
        """Test 1 scanner with previous position - should maintain direction."""
        scanner = mock_scanner("scanner1", (0.0, 0.0, 1.0), "Scanner1")
        mock_device._coordinator.devices = {"scanner1": scanner}
        mock_device.adverts = {"scanner1": mock_advert("scanner1", 10.0, 100.0)}
        mock_device.calculated_position = (10.0, 0.0, 1.0)  # 10m east of scanner

        result = calculate_position(mock_device, 101.0)

        assert result is not None
        assert result.scanner_count == 1
        assert result.method == "1-scanner-directional"
        # Should maintain eastward direction at new distance
        assert abs(result.x - 10.0) < 1.0
        assert abs(result.y - 0.0) < 1.0


class TestCalculatePosition2Scanners:
    """Test 2-scanner bilateration."""

    def test_2_scanners_valid_intersection(self, mock_device, mock_scanner, mock_advert):
        """Test 2 scanners with valid circle intersection."""
        scanner1 = mock_scanner("s1", (0.0, 0.0, 1.0), "Scanner1")
        scanner2 = mock_scanner("s2", (10.0, 0.0, 1.0), "Scanner2")
        mock_device._coordinator.devices = {"s1": scanner1, "s2": scanner2}
        mock_device.adverts = {
            "s1": mock_advert("s1", 5.0, 100.0),
            "s2": mock_advert("s2", 5.0, 100.0),
        }

        result = calculate_position(mock_device, 101.0)

        assert result is not None
        assert result.scanner_count == 2
        assert result.method in ["2-scanner-bilateration", "2-scanner-fallback"]
        # Should be roughly in the middle (5, y, 1)
        assert abs(result.x - 5.0) < 2.0

    def test_2_scanners_circles_dont_intersect(self, mock_device, mock_scanner, mock_advert):
        """Test 2 scanners where circles don't reach each other."""
        scanner1 = mock_scanner("s1", (0.0, 0.0, 1.0), "Scanner1")
        scanner2 = mock_scanner("s2", (100.0, 0.0, 1.0), "Scanner2")
        mock_device._coordinator.devices = {"s1": scanner1, "s2": scanner2}
        mock_device.adverts = {
            "s1": mock_advert("s1", 5.0, 100.0),  # d1 + d2 = 10 < 100
            "s2": mock_advert("s2", 5.0, 100.0),
        }

        result = calculate_position(mock_device, 101.0)

        # Should fall back to weighted centroid
        assert result is not None
        assert result.method == "2-scanner-fallback"

    def test_2_scanners_one_contains_other(self, mock_device, mock_scanner, mock_advert):
        """Test 2 scanners where one circle contains the other."""
        scanner1 = mock_scanner("s1", (0.0, 0.0, 1.0), "Scanner1")
        scanner2 = mock_scanner("s2", (5.0, 0.0, 1.0), "Scanner2")
        mock_device._coordinator.devices = {"s1": scanner1, "s2": scanner2}
        mock_device.adverts = {
            "s1": mock_advert("s1", 20.0, 100.0),  # |d1 - d2| = 18 > 5
            "s2": mock_advert("s2", 2.0, 100.0),
        }

        result = calculate_position(mock_device, 101.0)

        # Should fall back to weighted centroid
        assert result is not None
        assert result.method == "2-scanner-fallback"

    def test_2_scanners_with_velocity_constraint(self, mock_device, mock_scanner, mock_advert):
        """Test 2 scanners with previous position and velocity check."""
        scanner1 = mock_scanner("s1", (0.0, 0.0, 1.0), "Scanner1")
        scanner2 = mock_scanner("s2", (10.0, 0.0, 1.0), "Scanner2")
        mock_device._coordinator.devices = {"s1": scanner1, "s2": scanner2}
        mock_device.adverts = {
            "s1": mock_advert("s1", 5.0, 100.0),
            "s2": mock_advert("s2", 5.0, 100.0),
        }
        mock_device.calculated_position = (5.0, 3.0, 1.0)
        mock_device.position_timestamp = 100.0

        result = calculate_position(mock_device, 101.0)

        assert result is not None
        assert result.scanner_count == 2


class TestCalculatePosition3Scanners:
    """Test 3-scanner trilateration."""

    def test_3_scanners_weighted_centroid(self, mock_device, mock_scanner, mock_advert):
        """Test 3 scanners using weighted centroid."""
        scanner1 = mock_scanner("s1", (0.0, 0.0, 1.0), "Scanner1")
        scanner2 = mock_scanner("s2", (10.0, 0.0, 1.0), "Scanner2")
        scanner3 = mock_scanner("s3", (5.0, 10.0, 1.0), "Scanner3")
        mock_device._coordinator.devices = {
            "s1": scanner1,
            "s2": scanner2,
            "s3": scanner3,
        }
        mock_device.adverts = {
            "s1": mock_advert("s1", 7.0, 100.0),
            "s2": mock_advert("s2", 7.0, 100.0),
            "s3": mock_advert("s3", 7.0, 100.0),
        }

        result = calculate_position(mock_device, 101.0)

        assert result is not None
        assert result.scanner_count == 3
        assert result.method == "3-scanner"
        assert result.confidence > 50.0
        # Should be roughly in the center
        assert 2.0 < result.x < 8.0
        assert 2.0 < result.y < 8.0

    def test_3_scanners_unequal_distances(self, mock_device, mock_scanner, mock_advert):
        """Test 3 scanners with varying distances (higher weight to closer)."""
        scanner1 = mock_scanner("s1", (0.0, 0.0, 1.0), "Scanner1")
        scanner2 = mock_scanner("s2", (10.0, 0.0, 1.0), "Scanner2")
        scanner3 = mock_scanner("s3", (5.0, 10.0, 1.0), "Scanner3")
        mock_device._coordinator.devices = {
            "s1": scanner1,
            "s2": scanner2,
            "s3": scanner3,
        }
        mock_device.adverts = {
            "s1": mock_advert("s1", 2.0, 100.0),  # Very close
            "s2": mock_advert("s2", 20.0, 100.0),  # Far
            "s3": mock_advert("s3", 20.0, 100.0),  # Far
        }

        result = calculate_position(mock_device, 101.0)

        assert result is not None
        assert result.scanner_count == 3
        # Should be weighted toward scanner1 (closer)
        assert result.x < 5.0


class TestCalculatePosition4PlusScanners:
    """Test 4+ scanner overdetermined trilateration."""

    def test_4_scanners_high_confidence(self, mock_device, mock_scanner, mock_advert):
        """Test 4 scanners should give high confidence."""
        scanners = {
            "s1": mock_scanner("s1", (0.0, 0.0, 1.0), "Scanner1"),
            "s2": mock_scanner("s2", (10.0, 0.0, 1.0), "Scanner2"),
            "s3": mock_scanner("s3", (10.0, 10.0, 1.0), "Scanner3"),
            "s4": mock_scanner("s4", (0.0, 10.0, 1.0), "Scanner4"),
        }
        mock_device._coordinator.devices = scanners
        mock_device.adverts = {
            "s1": mock_advert("s1", 7.0, 100.0),
            "s2": mock_advert("s2", 7.0, 100.0),
            "s3": mock_advert("s3", 7.0, 100.0),
            "s4": mock_advert("s4", 7.0, 100.0),
        }

        result = calculate_position(mock_device, 101.0)

        assert result is not None
        assert result.scanner_count == 4
        assert result.method == "4+scanner"
        assert result.confidence > 65.0
        # Should be near center of square
        assert 3.0 < result.x < 7.0
        assert 3.0 < result.y < 7.0

    def test_5_scanners(self, mock_device, mock_scanner, mock_advert):
        """Test 5 scanners for even better accuracy."""
        scanners = {
            "s1": mock_scanner("s1", (0.0, 0.0, 1.0)),
            "s2": mock_scanner("s2", (10.0, 0.0, 1.0)),
            "s3": mock_scanner("s3", (10.0, 10.0, 1.0)),
            "s4": mock_scanner("s4", (0.0, 10.0, 1.0)),
            "s5": mock_scanner("s5", (5.0, 5.0, 2.0)),
        }
        mock_device._coordinator.devices = scanners
        mock_device.adverts = {
            k: mock_advert(k, 7.0, 100.0) for k in scanners.keys()
        }

        result = calculate_position(mock_device, 101.0)

        assert result is not None
        assert result.scanner_count == 5
        assert result.confidence >= 65.0


class TestCalculatePositionEdgeCases:
    """Test edge cases and error conditions."""

    def test_no_scanners(self, mock_device):
        """Test with no valid scanners."""
        result = calculate_position(mock_device, 100.0)
        assert result is None

    def test_scanner_without_position(self, mock_device, mock_scanner, mock_advert):
        """Test scanner without position configured."""
        scanner = mock_scanner("s1", None, "Scanner1")
        scanner.position = None
        mock_device._coordinator.devices = {"s1": scanner}
        mock_device.adverts = {"s1": mock_advert("s1", 5.0, 100.0)}

        result = calculate_position(mock_device, 101.0)
        assert result is None

    def test_invalid_distance(self, mock_device, mock_scanner, mock_advert):
        """Test with invalid (zero/negative) distance."""
        scanner = mock_scanner("s1", (10.0, 10.0, 1.0), "Scanner1")
        mock_device._coordinator.devices = {"s1": scanner}
        mock_device.adverts = {"s1": mock_advert("s1", 0.0, 100.0)}

        result = calculate_position(mock_device, 101.0)
        assert result is None

    def test_stale_advertisement(self, mock_device, mock_scanner, mock_advert):
        """Test with stale advertisement data."""
        scanner = mock_scanner("s1", (10.0, 10.0, 1.0), "Scanner1")
        mock_device._coordinator.devices = {"s1": scanner}
        # Advert is 100 seconds old (> TRILATERATION_POSITION_TIMEOUT)
        mock_device.adverts = {"s1": mock_advert("s1", 5.0, 0.0)}

        result = calculate_position(mock_device, 100.0)
        assert result is None

    def test_2_scanners_at_same_location(self, mock_device, mock_scanner, mock_advert):
        """Test degenerate case: 2 scanners at identical position.
        
        This tests the fix for ZeroDivisionError when scanner_distance == 0.0.
        Uses exact (0.0, 0.0, 0.0) positions to guarantee true zero distance.
        """
        scanner1 = mock_scanner("s1", (0.0, 0.0, 0.0), "Scanner1")
        scanner2 = mock_scanner("s2", (0.0, 0.0, 0.0), "Scanner2")
        mock_device._coordinator.devices = {"s1": scanner1, "s2": scanner2}
        mock_device.adverts = {
            "s1": mock_advert("s1", 3.0, 100.0),
            "s2": mock_advert("s2", 3.0, 100.0),
        }

        result = calculate_position(mock_device, 101.0)
        # Must return None (can't trilaterate with scanners at same position)
        assert result is None

    def test_high_variance_distances(self, mock_device, mock_scanner, mock_advert):
        """Test that high variance in distances reduces confidence."""
        scanners = {
            "s1": mock_scanner("s1", (0.0, 0.0, 1.0)),
            "s2": mock_scanner("s2", (10.0, 0.0, 1.0)),
            "s3": mock_scanner("s3", (5.0, 10.0, 1.0)),
        }
        mock_device._coordinator.devices = scanners
        mock_device.adverts = {
            "s1": mock_advert("s1", 1.0, 100.0),  # Very close
            "s2": mock_advert("s2", 50.0, 100.0),  # Very far
            "s3": mock_advert("s3", 25.0, 100.0),  # Medium
        }

        result = calculate_position(mock_device, 101.0)

        assert result is not None
        # Confidence should be reduced due to high variance
        # (This tests the variance penalty logic)
        assert result.confidence < 70.0


class TestTrilaterationResult:
    """Test TrilaterationResult dataclass."""

    def test_result_creation(self):
        """Test creating a result object."""
        result = TrilaterationResult(
            x=5.0,
            y=10.0,
            z=1.5,
            confidence=75.0,
            scanner_count=3,
            method="3-scanner",
        )
        assert result.x == 5.0
        assert result.y == 10.0
        assert result.z == 1.5
        assert result.confidence == 75.0
        assert result.scanner_count == 3
        assert result.method == "3-scanner"
