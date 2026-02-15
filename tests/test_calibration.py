"""Tests for auto-calibration utilities."""

import math

import pytest

from custom_components.bermuda.util import (
    calculate_distance_from_position,
    calculate_scanner_offsets_from_scanner_pairs,
    calculate_scanner_rssi_offsets,
    get_calibration_quality_metrics,
    get_scanner_pair_quality_metrics,
)


def test_calculate_distance_from_position():
    """Test 3D distance calculation."""
    # Test same position
    assert calculate_distance_from_position((0, 0, 0), (0, 0, 0)) == 0.0

    # Test 1m in X direction
    assert calculate_distance_from_position((0, 0, 0), (1, 0, 0)) == pytest.approx(1.0)

    # Test 1m in Y direction
    assert calculate_distance_from_position((0, 0, 0), (0, 1, 0)) == pytest.approx(1.0)

    # Test 1m in Z direction
    assert calculate_distance_from_position((0, 0, 0), (0, 0, 1)) == pytest.approx(1.0)

    # Test 3D Pythagorean: 3-4-5 triangle
    assert calculate_distance_from_position((0, 0, 0), (3, 4, 0)) == pytest.approx(5.0)

    # Test 3D diagonal: sqrt(3)
    assert calculate_distance_from_position((0, 0, 0), (1, 1, 1)) == pytest.approx(math.sqrt(3))


def test_calculate_scanner_rssi_offsets_basic():
    """Test basic offset calculation."""
    # Setup: Two scanners at different positions
    scanner_positions = {
        "aa:aa:aa:aa:aa:aa": (0.0, 0.0, 0.0),  # Scanner at origin
        "bb:bb:bb:bb:bb:bb": (5.0, 0.0, 0.0),  # Scanner 5m away in X
    }

    # Beacon at 1m from first scanner (distance = 1m)
    # and sqrt(17) ≈ 4.12m from second scanner
    target_position = (1.0, 0.0, 0.0)

    # Simulate RSSI data
    # Using ref_power=-65, attenuation=2.7
    # Expected RSSI at 1m: -65 dBm
    # Expected RSSI at 4.12m: -65 - 10*2.7*log10(4.12) ≈ -65 - 16.5 = -81.5 dBm

    ref_power = -65.0
    attenuation = 2.7

    # Scanner AA observes -65 dBm (perfect match)
    # Scanner BB observes -81.5 dBm (perfect match)
    scanner_rssi_data = {
        "aa:aa:aa:aa:aa:aa": [-65.0] * 20,  # 20 samples at -65 dBm
        "bb:bb:bb:bb:bb:bb": [-81.5] * 20,  # 20 samples at -81.5 dBm
    }

    offsets = calculate_scanner_rssi_offsets(
        scanner_rssi_data,
        scanner_positions,
        target_position,
        ref_power,
        attenuation,
        mode="offsets_only",
    )

    # Both scanners should have ~0 offset since they match the model perfectly
    assert offsets["aa:aa:aa:aa:aa:aa"] == pytest.approx(0.0, abs=0.1)
    assert offsets["bb:bb:bb:bb:bb:bb"] == pytest.approx(0.0, abs=0.1)


def test_calculate_scanner_rssi_offsets_with_bias():
    """Test offset calculation when scanners have sensitivity bias."""
    scanner_positions = {
        "aa:aa:aa:aa:aa:aa": (0.0, 0.0, 0.0),
        "bb:bb:bb:bb:bb:bb": (5.0, 0.0, 0.0),
    }

    target_position = (1.0, 0.0, 0.0)
    ref_power = -65.0
    attenuation = 2.7

    # Scanner AA observes -70 dBm (5 dB weaker than expected -65)
    # Scanner BB observes -76.5 dBm (5 dB stronger than expected -81.5)
    scanner_rssi_data = {
        "aa:aa:aa:aa:aa:aa": [-70.0] * 20,  # Weak receiver
        "bb:bb:bb:bb:bb:bb": [-76.5] * 20,  # Strong receiver
    }

    offsets = calculate_scanner_rssi_offsets(
        scanner_rssi_data,
        scanner_positions,
        target_position,
        ref_power,
        attenuation,
        mode="offsets_only",
    )

    # Offset = expected - observed
    # AA: -65 - (-70) = +5 dB (boost weak receiver)
    # BB: -81.5 - (-76.5) = -5 dB (attenuate strong receiver)
    assert offsets["aa:aa:aa:aa:aa:aa"] == pytest.approx(5.0, abs=0.1)
    assert offsets["bb:bb:bb:bb:bb:bb"] == pytest.approx(-5.0, abs=0.1)


def test_calculate_scanner_rssi_offsets_insufficient_samples():
    """Test that scanners with too few samples are excluded."""
    scanner_positions = {
        "aa:aa:aa:aa:aa:aa": (0.0, 0.0, 0.0),
        "bb:bb:bb:bb:bb:bb": (5.0, 0.0, 0.0),
    }

    target_position = (1.0, 0.0, 0.0)

    # Only 5 samples (less than the minimum 10)
    scanner_rssi_data = {
        "aa:aa:aa:aa:aa:aa": [-65.0] * 5,
        "bb:bb:bb:bb:bb:bb": [-81.5] * 20,
    }

    offsets = calculate_scanner_rssi_offsets(
        scanner_rssi_data,
        scanner_positions,
        target_position,
        -65.0,
        2.7,
        mode="offsets_only",
    )

    # AA should be excluded, only BB should have offset
    assert "aa:aa:aa:aa:aa:aa" not in offsets
    assert "bb:bb:bb:bb:bb:bb" in offsets


def test_calculate_scanner_rssi_offsets_insufficient_scanners():
    """Test that calibration fails with less than 2 scanners."""
    scanner_positions = {
        "aa:aa:aa:aa:aa:aa": (0.0, 0.0, 0.0),
    }

    target_position = (1.0, 0.0, 0.0)

    scanner_rssi_data = {
        "aa:aa:aa:aa:aa:aa": [-65.0] * 20,
    }

    offsets = calculate_scanner_rssi_offsets(
        scanner_rssi_data,
        scanner_positions,
        target_position,
        -65.0,
        2.7,
        mode="offsets_only",
    )

    # Should return empty dict (insufficient scanners)
    assert offsets == {}


def test_calculate_scanner_rssi_offsets_very_close_beacon():
    """Test that very close beacons (< 0.1m) are handled gracefully."""
    scanner_positions = {
        "aa:aa:aa:aa:aa:aa": (0.0, 0.0, 0.0),
        "bb:bb:bb:bb:bb:bb": (5.0, 0.0, 0.0),
    }

    # Beacon very close to scanner AA (0.05m)
    target_position = (0.05, 0.0, 0.0)

    scanner_rssi_data = {
        "aa:aa:aa:aa:aa:aa": [-50.0] * 20,
        "bb:bb:bb:bb:bb:bb": [-81.5] * 20,
    }

    offsets = calculate_scanner_rssi_offsets(
        scanner_rssi_data,
        scanner_positions,
        target_position,
        -65.0,
        2.7,
        mode="offsets_only",
    )

    # Scanner AA should be excluded (too close), only BB should have offset
    assert "aa:aa:aa:aa:aa:aa" not in offsets
    assert "bb:bb:bb:bb:bb:bb" in offsets


def test_get_calibration_quality_metrics():
    """Test calibration quality metrics calculation."""
    scanner_positions = {
        "aa:aa:aa:aa:aa:aa": (0.0, 0.0, 0.0),
        "bb:bb:bb:bb:bb:bb": (5.0, 0.0, 0.0),
    }

    target_position = (1.0, 0.0, 0.0)

    # Scanner AA: stable readings around -65
    # Scanner BB: noisy readings around -81
    scanner_rssi_data = {
        "aa:aa:aa:aa:aa:aa": [-65.0, -65.0, -65.0, -65.0, -65.0],
        "bb:bb:bb:bb:bb:bb": [-81.0, -82.0, -80.0, -83.0, -81.0],
    }

    metrics = get_calibration_quality_metrics(
        scanner_rssi_data,
        scanner_positions,
        target_position,
    )

    # Check AA metrics
    assert metrics["aa:aa:aa:aa:aa:aa"]["sample_count"] == 5
    assert metrics["aa:aa:aa:aa:aa:aa"]["rssi_median"] == pytest.approx(-65.0)
    assert metrics["aa:aa:aa:aa:aa:aa"]["rssi_std"] == pytest.approx(0.0)  # Perfectly stable
    assert metrics["aa:aa:aa:aa:aa:aa"]["distance"] == pytest.approx(1.0)

    # Check BB metrics
    assert metrics["bb:bb:bb:bb:bb:bb"]["sample_count"] == 5
    assert metrics["bb:bb:bb:bb:bb:bb"]["rssi_median"] == pytest.approx(-81.0)
    assert metrics["bb:bb:bb:bb:bb:bb"]["rssi_std"] > 0  # Has variance
    assert metrics["bb:bb:bb:bb:bb:bb"]["distance"] == pytest.approx(4.0)


def test_calculate_scanner_rssi_offsets_unsupported_mode():
    """Test that unsupported calibration mode raises ValueError."""
    scanner_positions = {
        "aa:aa:aa:aa:aa:aa": (0.0, 0.0, 0.0),
    }

    target_position = (1.0, 0.0, 0.0)

    scanner_rssi_data = {
        "aa:aa:aa:aa:aa:aa": [-65.0] * 20,
    }

    with pytest.raises(ValueError, match="Unsupported calibration mode"):
        calculate_scanner_rssi_offsets(
            scanner_rssi_data,
            scanner_positions,
            target_position,
            -65.0,
            2.7,
            mode="invalid_mode",
        )


def test_calculate_scanner_rssi_offsets_full_mode_not_implemented():
    """Test that full calibration mode is not yet implemented."""
    scanner_positions = {
        "aa:aa:aa:aa:aa:aa": (0.0, 0.0, 0.0),
        "bb:bb:bb:bb:bb:bb": (5.0, 0.0, 0.0),
    }

    target_position = (1.0, 0.0, 0.0)

    scanner_rssi_data = {
        "aa:aa:aa:aa:aa:aa": [-65.0] * 20,
        "bb:bb:bb:bb:bb:bb": [-81.5] * 20,
    }

    with pytest.raises(NotImplementedError, match="Full calibration mode"):
        calculate_scanner_rssi_offsets(
            scanner_rssi_data,
            scanner_positions,
            target_position,
            -65.0,
            2.7,
            mode="full",
        )


def test_calculate_scanner_offsets_from_scanner_pairs_basic():
    """Test basic scanner-to-scanner calibration."""
    # Setup: 3 scanners in a line
    scanner_positions = {
        "aa:aa:aa:aa:aa:aa": (0.0, 0.0, 0.0),
        "bb:bb:bb:bb:bb:bb": (5.0, 0.0, 0.0),
        "cc:cc:cc:cc:cc:cc": (10.0, 0.0, 0.0),
    }

    # Scanner pairs with perfect RSSI measurements
    # ref_power=-65, attenuation=2.7
    # Distance AA-BB: 5m, expected RSSI: -65 - 10*2.7*log10(5) ≈ -83.9 dBm
    # Distance BB-CC: 5m, expected RSSI: -83.9 dBm
    # Distance AA-CC: 10m, expected RSSI: -65 - 10*2.7*log10(10) = -92.0 dBm

    scanner_pairs_rssi = {
        ("bb:bb:bb:bb:bb:bb", "aa:aa:aa:aa:aa:aa"): [-83.9] * 20,  # BB receives from AA
        ("aa:aa:aa:aa:aa:aa", "bb:bb:bb:bb:bb:bb"): [-83.9] * 20,  # AA receives from BB
        ("cc:cc:cc:cc:cc:cc", "bb:bb:bb:bb:bb:bb"): [-83.9] * 20,  # CC receives from BB
        ("bb:bb:bb:bb:bb:bb", "cc:cc:cc:cc:cc:cc"): [-83.9] * 20,  # BB receives from CC
        ("cc:cc:cc:cc:cc:cc", "aa:aa:aa:aa:aa:aa"): [-92.0] * 20,  # CC receives from AA
        ("aa:aa:aa:aa:aa:aa", "cc:cc:cc:cc:cc:cc"): [-92.0] * 20,  # AA receives from CC
    }

    offsets = calculate_scanner_offsets_from_scanner_pairs(
        scanner_pairs_rssi,
        scanner_positions,
        ref_power=-65.0,
        attenuation=2.7,
    )

    # Reference scanner should have offset 0
    assert "aa:aa:aa:aa:aa:aa" in offsets
    assert offsets["aa:aa:aa:aa:aa:aa"] == pytest.approx(0.0, abs=0.1)

    # Other scanners should also have ~0 offset (perfect measurements)
    assert "bb:bb:bb:bb:bb:bb" in offsets
    assert offsets["bb:bb:bb:bb:bb:bb"] == pytest.approx(0.0, abs=0.5)

    assert "cc:cc:cc:cc:cc:cc" in offsets
    assert offsets["cc:cc:cc:cc:cc:cc"] == pytest.approx(0.0, abs=0.5)


def test_calculate_scanner_offsets_from_scanner_pairs_with_bias():
    """Test scanner-to-scanner calibration with scanner bias."""
    scanner_positions = {
        "aa:aa:aa:aa:aa:aa": (0.0, 0.0, 0.0),
        "bb:bb:bb:bb:bb:bb": (5.0, 0.0, 0.0),
    }

    # AA is reference (offset 0)
    # BB has offset +5 dB
    # AA receives from BB: expected -83.9, observed = expected - offset_BB + offset_AA = -83.9 - 5 + 0 = -88.9
    # BB receives from AA: expected -83.9, observed = expected - offset_AA + offset_BB = -83.9 - 0 + 5 = -78.9

    scanner_pairs_rssi = {
        ("aa:aa:aa:aa:aa:aa", "bb:bb:bb:bb:bb:bb"): [-88.9] * 20,  # AA receives from BB (weaker)
        ("bb:bb:bb:bb:bb:bb", "aa:aa:aa:aa:aa:aa"): [-78.9] * 20,  # BB receives from AA (stronger)
    }

    offsets = calculate_scanner_offsets_from_scanner_pairs(
        scanner_pairs_rssi,
        scanner_positions,
        ref_power=-65.0,
        attenuation=2.7,
    )

    # Reference scanner
    assert offsets["aa:aa:aa:aa:aa:aa"] == pytest.approx(0.0)

    # BB should have positive offset
    assert offsets["bb:bb:bb:bb:bb:bb"] == pytest.approx(5.0, abs=0.5)


def test_calculate_scanner_offsets_from_scanner_pairs_insufficient_data():
    """Test that insufficient scanner pairs returns empty dict."""
    scanner_positions = {
        "aa:aa:aa:aa:aa:aa": (0.0, 0.0, 0.0),
    }

    scanner_pairs_rssi = {}

    offsets = calculate_scanner_offsets_from_scanner_pairs(
        scanner_pairs_rssi,
        scanner_positions,
        ref_power=-65.0,
        attenuation=2.7,
    )

    assert offsets == {}


def test_get_scanner_pair_quality_metrics():
    """Test scanner pair quality metrics calculation."""
    scanner_positions = {
        "aa:aa:aa:aa:aa:aa": (0.0, 0.0, 0.0),
        "bb:bb:bb:bb:bb:bb": (5.0, 0.0, 0.0),
    }

    scanner_pairs_rssi = {
        ("aa:aa:aa:aa:aa:aa", "bb:bb:bb:bb:bb:bb"): [-83.0, -84.0, -83.0, -85.0, -83.0],
        ("bb:bb:bb:bb:bb:bb", "aa:aa:aa:aa:aa:aa"): [-84.0, -84.0, -84.0, -84.0, -84.0],
    }

    metrics = get_scanner_pair_quality_metrics(scanner_pairs_rssi, scanner_positions)

    # Check first pair (AA receives from BB)
    pair1 = ("aa:aa:aa:aa:aa:aa", "bb:bb:bb:bb:bb:bb")
    assert pair1 in metrics
    assert metrics[pair1]["sample_count"] == 5
    assert metrics[pair1]["rssi_median"] == pytest.approx(-83.0)
    assert metrics[pair1]["rssi_std"] > 0  # Has variance
    assert metrics[pair1]["distance"] == pytest.approx(5.0)

    # Check second pair (BB receives from AA)
    pair2 = ("bb:bb:bb:bb:bb:bb", "aa:aa:aa:aa:aa:aa")
    assert pair2 in metrics
    assert metrics[pair2]["sample_count"] == 5
    assert metrics[pair2]["rssi_median"] == pytest.approx(-84.0)
    assert metrics[pair2]["rssi_std"] == pytest.approx(0.0)  # Perfectly stable
    assert metrics[pair2]["distance"] == pytest.approx(5.0)
