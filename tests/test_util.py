"""Test util.py in Bermuda."""

from __future__ import annotations

# from homeassistant.core import HomeAssistant

from math import floor
from unittest.mock import Mock

from custom_components.bermuda import util


def test_mac_math_offset():
    assert util.mac_math_offset("aa:bb:cc:dd:ee:ef", 2) == "aa:bb:cc:dd:ee:f1"
    assert util.mac_math_offset("aa:bb:cc:dd:ee:ef", -3) == "aa:bb:cc:dd:ee:ec"
    assert util.mac_math_offset("aa:bb:cc:dd:ee:ff", 2) is None
    assert util.mac_math_offset("clearly_not:a-mac_address", 2) == None
    assert util.mac_math_offset(None, 4) == None


def test_mac_norm():
    assert util.mac_norm("AA:bb:CC:88:Ff:00") == "aa:bb:cc:88:ff:00"
    assert util.mac_norm("Not_exactly-a-MAC:address") == "not_exactly-a-mac:address"
    assert util.mac_norm("aa_bb_CC_dd_ee_ff") == "aa:bb:cc:dd:ee:ff"
    assert util.mac_norm("aa-77-CC-dd-ee-ff") == "aa:77:cc:dd:ee:ff"


def test_mac_explode_formats():
    ex = util.mac_explode_formats("aa:bb:cc:77:ee:ff")
    assert "aa:bb:cc:77:ee:ff" in ex
    assert "aa-bb-cc-77-ee-ff" in ex
    for e in ex:
        assert len(e) in [12, 17]


def test_mac_redact():
    assert util.mac_redact("aa:bb:cc:77:ee:ff", "tEstMe") == "aa::tEstMe::ff"
    assert util.mac_redact("howdy::doody::friend", "PLEASENOE") == "ho::PLEASENOE::nd"


def test_rssi_to_metres():
    assert floor(util.rssi_to_metres(-50, -20, 2)) == 31
    assert floor(util.rssi_to_metres(-80, -20, 2)) == 1000


def test_clean_charbuf():
    assert util.clean_charbuf("a Normal string.") == "a Normal string."
    assert util.clean_charbuf("Broken\000String\000Fixed\000\000\000") == "Broken"


def test_validate_scanners_for_trilateration():
    """Test scanner validation helper function."""

    # Create mock coordinator
    mock_coordinator = Mock()
    mock_coordinator.devices = {}

    # Create mock device with adverts
    mock_device = Mock()
    mock_device._coordinator = mock_coordinator
    mock_device.adverts = {}

    # Test 1: Empty adverts - should return empty list
    result = util.validate_scanners_for_trilateration(mock_device, 100.0, 30.0)
    assert result == []

    # Test 2: Scanner not in devices dict
    mock_advert1 = Mock()
    mock_advert1.scanner_address = "aa:bb:cc:dd:ee:01"
    mock_advert1.stamp = 95.0
    mock_advert1.rssi_distance = 5.0
    mock_device.adverts = {"scanner1": mock_advert1}

    result = util.validate_scanners_for_trilateration(mock_device, 100.0, 30.0)
    assert result == []

    # Test 3: Scanner exists but is_scanner=False
    mock_scanner1 = Mock()
    mock_scanner1.is_scanner = False
    mock_scanner1.name = "Not a scanner"
    mock_coordinator.devices["aa:bb:cc:dd:ee:01"] = mock_scanner1

    result = util.validate_scanners_for_trilateration(mock_device, 100.0, 30.0)
    assert result == []

    # Test 4: Scanner exists, is_scanner=True, but no position
    mock_scanner1.is_scanner = True
    mock_scanner1.position = None

    result = util.validate_scanners_for_trilateration(mock_device, 100.0, 30.0)
    assert result == []

    # Test 5: Scanner valid but rssi_distance is None
    mock_scanner1.position = (1.0, 2.0, 3.0)
    mock_advert1.rssi_distance = None

    result = util.validate_scanners_for_trilateration(mock_device, 100.0, 30.0)
    assert result == []

    # Test 6: Scanner valid but rssi_distance <= 0
    mock_advert1.rssi_distance = 0.0

    result = util.validate_scanners_for_trilateration(mock_device, 100.0, 30.0)
    assert result == []

    mock_advert1.rssi_distance = -1.5

    result = util.validate_scanners_for_trilateration(mock_device, 100.0, 30.0)
    assert result == []

    # Test 7: Scanner valid but advert is stale (too old)
    mock_advert1.rssi_distance = 5.0
    mock_advert1.stamp = 60.0  # 40 seconds old (current_time=100, timeout=30)

    result = util.validate_scanners_for_trilateration(mock_device, 100.0, 30.0)
    assert result == []

    # Test 8: Valid scanner - all checks pass
    mock_advert1.stamp = 95.0  # 5 seconds old (within timeout)

    result = util.validate_scanners_for_trilateration(mock_device, 100.0, 30.0)
    assert len(result) == 1
    assert result[0] == (mock_scanner1, mock_advert1)

    # Test 9: Multiple scanners - mixed valid/invalid
    mock_advert2 = Mock()
    mock_advert2.scanner_address = "aa:bb:cc:dd:ee:02"
    mock_advert2.stamp = 80.0  # 20 seconds old
    mock_advert2.rssi_distance = 3.5

    mock_scanner2 = Mock()
    mock_scanner2.is_scanner = True
    mock_scanner2.position = (4.0, 5.0, 1.5)
    mock_scanner2.name = "Scanner 2"
    mock_coordinator.devices["aa:bb:cc:dd:ee:02"] = mock_scanner2

    mock_advert3 = Mock()
    mock_advert3.scanner_address = "aa:bb:cc:dd:ee:03"
    mock_advert3.stamp = 50.0  # 50 seconds old - stale!
    mock_advert3.rssi_distance = 7.0

    mock_scanner3 = Mock()
    mock_scanner3.is_scanner = True
    mock_scanner3.position = (7.0, 8.0, 2.0)
    mock_scanner3.name = "Scanner 3 (stale)"
    mock_coordinator.devices["aa:bb:cc:dd:ee:03"] = mock_scanner3

    mock_device.adverts = {
        "scanner1": mock_advert1,
        "scanner2": mock_advert2,
        "scanner3": mock_advert3,
    }

    result = util.validate_scanners_for_trilateration(mock_device, 100.0, 30.0)
    assert len(result) == 2  # scanner1 and scanner2 valid, scanner3 stale
    assert (mock_scanner1, mock_advert1) in result
    assert (mock_scanner2, mock_advert2) in result
    assert (mock_scanner3, mock_advert3) not in result

    # Test 10: Edge case - advert exactly at timeout boundary
    mock_advert1.stamp = 70.0  # Exactly 30 seconds old
    mock_device.adverts = {"scanner1": mock_advert1}

    result = util.validate_scanners_for_trilateration(mock_device, 100.0, 30.0)
    assert len(result) == 1  # Should still be valid (not greater than timeout)
