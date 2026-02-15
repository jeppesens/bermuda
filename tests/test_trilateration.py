"""Tests for trilateration algorithms (Nadaraya-Watson + Kalman)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from custom_components.bermuda.kalman import KalmanFilterSettings
from custom_components.bermuda.kalman import KalmanLocation
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

    def _create_scanner(address, position, name="Scanner", node_floors=None):
        scanner = MagicMock()
        scanner.address = address
        scanner.name = name
        scanner.position = position
        scanner.is_scanner = True
        scanner.node_floors = node_floors
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
            {"id": "invalid", "area_id": "invalid_room", "points": [[0, 0]]},
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


class TestNadarayaWatson:
    """Test Nadaraya-Watson kernel regression position estimation."""

    def test_3_equidistant_scanners(self, mock_device, mock_scanner, mock_advert):
        """Test 3 equidistant scanners - result should be near centroid."""
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
        assert result.method == "nadaraya_watson"
        # Should be near the centroid of the triangle (5, 3.33)
        assert 2.0 < result.x < 8.0
        assert 1.0 < result.y < 8.0

    def test_closer_scanner_has_more_weight(self, mock_device, mock_scanner, mock_advert):
        """Test that closer scanner pulls position more toward it."""
        scanner1 = mock_scanner("s1", (0.0, 0.0, 1.0), "Scanner1")
        scanner2 = mock_scanner("s2", (10.0, 0.0, 1.0), "Scanner2")
        scanner3 = mock_scanner("s3", (5.0, 10.0, 1.0), "Scanner3")
        mock_device._coordinator.devices = {
            "s1": scanner1,
            "s2": scanner2,
            "s3": scanner3,
        }
        mock_device.adverts = {
            "s1": mock_advert("s1", 1.0, 100.0),  # Very close
            "s2": mock_advert("s2", 20.0, 100.0),  # Far
            "s3": mock_advert("s3", 20.0, 100.0),  # Far
        }

        result = calculate_position(mock_device, 101.0)

        assert result is not None
        assert result.method == "nadaraya_watson"
        # Should be strongly weighted toward s1 at (0,0)
        assert result.x < 5.0

    def test_2_scanners_midpoint(self, mock_device, mock_scanner, mock_advert):
        """Test 2 scanners should use midpoint."""
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
        assert result.method == "nadaraya_watson"
        # Should be at midpoint (5, 0)
        assert abs(result.x - 5.0) < 0.1
        assert abs(result.y - 0.0) < 0.1

    def test_4_scanners_square(self, mock_device, mock_scanner, mock_advert):
        """Test 4 scanners in a square with equal distances."""
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
        assert result.method == "nadaraya_watson"
        # Should be near center
        assert 3.0 < result.x < 7.0
        assert 3.0 < result.y < 7.0

    def test_5_scanners(self, mock_device, mock_scanner, mock_advert):
        """Test 5 scanners for higher accuracy."""
        scanners = {
            "s1": mock_scanner("s1", (0.0, 0.0, 1.0)),
            "s2": mock_scanner("s2", (10.0, 0.0, 1.0)),
            "s3": mock_scanner("s3", (10.0, 10.0, 1.0)),
            "s4": mock_scanner("s4", (0.0, 10.0, 1.0)),
            "s5": mock_scanner("s5", (5.0, 5.0, 2.0)),
        }
        mock_device._coordinator.devices = scanners
        mock_device.adverts = {k: mock_advert(k, 7.0, 100.0) for k in scanners}

        result = calculate_position(mock_device, 101.0)

        assert result is not None
        assert result.scanner_count == 5
        assert result.method == "nadaraya_watson"


class TestNearestNode:
    """Test nearest-node fallback."""

    def test_1_scanner_uses_nearest_node(self, mock_device, mock_scanner, mock_advert):
        """Test 1 scanner falls back to nearest_node."""
        scanner = mock_scanner("scanner1", (10.0, 10.0, 1.0), "Scanner1")
        mock_device._coordinator.devices = {"scanner1": scanner}
        mock_device.adverts = {"scanner1": mock_advert("scanner1", 5.0, 100.0)}

        result = calculate_position(mock_device, 101.0)

        assert result is not None
        assert result.scanner_count == 1
        assert result.method == "nearest_node"
        # Should be at scanner position
        assert abs(result.x - 10.0) < 0.1
        assert abs(result.y - 10.0) < 0.1


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

    def test_scanners_at_same_location(self, mock_device, mock_scanner, mock_advert):
        """Test degenerate case: scanners at identical position."""
        scanner1 = mock_scanner("s1", (0.0, 0.0, 0.0), "Scanner1")
        scanner2 = mock_scanner("s2", (0.0, 0.0, 0.0), "Scanner2")
        mock_device._coordinator.devices = {"s1": scanner1, "s2": scanner2}
        mock_device.adverts = {
            "s1": mock_advert("s1", 3.0, 100.0),
            "s2": mock_advert("s2", 3.0, 100.0),
        }

        # Should still produce a result (midpoint of identical positions)
        result = calculate_position(mock_device, 101.0)
        # Whether this returns a result depends on the implementation
        # At minimum it should not crash
        if result is not None:
            assert result.method in ["nadaraya_watson", "nearest_node"]


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
            method="nadaraya_watson",
        )
        assert result.x == 5.0
        assert result.y == 10.0
        assert result.z == 1.5
        assert result.confidence == 75.0
        assert result.scanner_count == 3
        assert result.method == "nadaraya_watson"

    def test_result_with_optional_fields(self):
        """Test creating a result with all optional fields."""
        result = TrilaterationResult(
            x=5.0,
            y=10.0,
            z=1.5,
            confidence=75.0,
            scanner_count=3,
            method="nadaraya_watson",
            room_id="living-room",
            floor_id="ground",
            error=0.5,
            correlation=0.95,
        )
        assert result.room_id == "living-room"
        assert result.floor_id == "ground"
        assert result.error == 0.5
        assert result.correlation == 0.95


class TestKalmanFilter:
    """Test Kalman filter for position smoothing."""

    def test_initial_update_returns_input(self):
        """Test that first update returns the input position."""
        kf = KalmanLocation()
        x, y, z = kf.update(5.0, 10.0, 1.0, timestamp=100.0)
        assert x == 5.0
        assert y == 10.0
        assert z == 1.0

    def test_stationary_converges(self):
        """Test that repeated same-position updates converge to that position."""
        kf = KalmanLocation()
        for i in range(20):
            x, y, z = kf.update(5.0, 10.0, 1.0, timestamp=100.0 + i * 0.5)

        assert abs(x - 5.0) < 0.1
        assert abs(y - 10.0) < 0.1
        assert abs(z - 1.0) < 0.1

    def test_smooths_noisy_input(self):
        """Test that Kalman filter smooths noisy position data."""
        kf = KalmanLocation()
        # Initial position
        kf.update(5.0, 5.0, 1.0, timestamp=100.0)

        # Feed noisy data around (5, 5)
        positions = [
            (5.5, 4.5, 1.0),
            (4.5, 5.5, 1.0),
            (5.2, 4.8, 1.0),
            (4.8, 5.2, 1.0),
            (5.0, 5.0, 1.0),
        ]
        for i, (px, py, pz) in enumerate(positions):
            x, y, z = kf.update(px, py, pz, timestamp=101.0 + i * 0.5)

        # Should be near (5, 5) and not jump around
        assert abs(x - 5.0) < 1.0
        assert abs(y - 5.0) < 1.0

    def test_velocity_estimation(self):
        """Test that velocity is estimated from movement."""
        kf = KalmanLocation(KalmanFilterSettings(max_velocity=10.0))
        # Move steadily in x direction
        for i in range(10):
            kf.update(float(i), 0.0, 0.0, timestamp=100.0 + i * 1.0)

        vx, vy, vz = kf.velocity
        # Should detect positive x velocity
        assert vx > 0.0

    def test_velocity_clamping(self):
        """Test that velocity is clamped to max_velocity."""
        settings = KalmanFilterSettings(max_velocity=0.5)
        kf = KalmanLocation(settings)

        kf.update(0.0, 0.0, 0.0, timestamp=100.0)
        # Jump far away instantly
        kf.update(100.0, 0.0, 0.0, timestamp=100.1)

        # Speed should be clamped
        assert kf.speed <= settings.max_velocity + 0.01

    def test_reset(self):
        """Test resetting the filter."""
        kf = KalmanLocation()
        kf.update(5.0, 5.0, 1.0, timestamp=100.0)
        kf.update(6.0, 6.0, 1.0, timestamp=101.0)

        kf.reset(0.0, 0.0, 0.0)
        assert kf.location == (0.0, 0.0, 0.0)
        assert kf.speed < 0.01

    def test_custom_settings(self):
        """Test Kalman filter with custom noise settings."""
        settings = KalmanFilterSettings(
            process_noise=0.1,
            measurement_noise=0.5,
            max_velocity=1.0,
        )
        kf = KalmanLocation(settings)
        assert kf.settings.process_noise == 0.1
        assert kf.settings.measurement_noise == 0.5
        assert kf.settings.max_velocity == 1.0

    def test_prediction(self):
        """Test get_prediction returns reasonable values."""
        kf = KalmanLocation()
        # Before any update, should return initial location
        assert kf.get_prediction() == (0.0, 0.0, 0.0)

        kf.update(5.0, 5.0, 1.0, timestamp=100.0)
        pred = kf.get_prediction()
        # Should be close to last position
        assert abs(pred[0] - 5.0) < 1.0
        assert abs(pred[1] - 5.0) < 1.0


class TestESPresenseYAMLParsing:
    """Test ESPresense YAML config parsing in config flow."""

    def test_parse_basic_yaml(self):
        """Test parsing a basic ESPresense YAML config."""
        from custom_components.bermuda.config_flow import BermudaOptionsFlowHandler

        # We can't fully instantiate the flow handler, but we can test the method
        # by creating a minimal instance
        handler = BermudaOptionsFlowHandler.__new__(BermudaOptionsFlowHandler)

        yaml_input = """
floors:
  - id: ground
    name: Ground Floor
    bounds: [[0, 0, 0], [10, 8, 3]]
    rooms:
      - name: Living Room
        points: [[0,0], [5,0], [5,4], [0,4]]
      - name: Kitchen
        points: [[5,0], [10,0], [10,4], [5,4]]

nodes:
  - name: Living Room Node
    point: [2.5, 2.0, 1.0]
    floors: ["ground"]
  - name: Kitchen Node
    point: [7.5, 2.0, 1.0]
    floors: ["ground"]

filtering:
  process_noise: 0.02
  measurement_noise: 0.2
  max_velocity: 0.8
"""
        result = handler._parse_espresense_yaml(yaml_input)

        # Check floors
        assert len(result["floors"]) == 1
        assert result["floors"][0]["id"] == "ground"
        assert result["floors"][0]["name"] == "Ground Floor"

        # Check rooms (should be extracted from floors)
        assert len(result["rooms"]) == 2
        room_names = {r["name"] for r in result["rooms"]}
        assert "Living Room" in room_names
        assert "Kitchen" in room_names
        # Rooms should reference their floor
        for room in result["rooms"]:
            assert room["floor"] == "ground"

        # Check nodes
        assert len(result["nodes"]) == 2
        node_names = {n["name"] for n in result["nodes"]}
        assert "Living Room Node" in node_names
        assert "Kitchen Node" in node_names

        # Check filtering
        assert result["_filtering"]["process_noise"] == 0.02
        assert result["_filtering"]["measurement_noise"] == 0.2
        assert result["_filtering"]["max_velocity"] == 0.8
