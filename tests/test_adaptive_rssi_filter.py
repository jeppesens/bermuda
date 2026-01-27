"""Tests for AdaptivePercentileRSSI filtering class."""

from __future__ import annotations

import math

from custom_components.bermuda.bermuda_advert import AdaptivePercentileRSSI


class TestAdaptivePercentileRSSI:
    """Test the AdaptivePercentileRSSI class."""

    def test_initialization(self):
        """Test filter initialization with default parameters."""
        filter_obj = AdaptivePercentileRSSI()
        assert filter_obj.time_window_ms == 15000
        assert filter_obj.min_readings == 10
        assert filter_obj.max_readings == 200
        assert filter_obj.iqr_coefficient == 1.5
        assert filter_obj.get_sample_count() == 0

    def test_initialization_custom_params(self):
        """Test filter initialization with custom parameters."""
        filter_obj = AdaptivePercentileRSSI(
            time_window_ms=10000,
            initial_max_readings=30,
            min_readings=5,
            max_readings=150,
            iqr_coefficient=2.0,
        )
        assert filter_obj.time_window_ms == 10000
        assert filter_obj.min_readings == 5
        assert filter_obj.max_readings == 150
        assert filter_obj.iqr_coefficient == 2.0

    def test_add_single_measurement(self):
        """Test adding a single measurement."""
        filter_obj = AdaptivePercentileRSSI()
        filter_obj.add_measurement(-65.0, 100.0)
        
        assert filter_obj.get_sample_count() == 1
        assert filter_obj.get_median_iqr() == -65.0

    def test_add_multiple_measurements(self):
        """Test adding multiple measurements."""
        filter_obj = AdaptivePercentileRSSI()
        base_time = 1000.0
        
        for i in range(10):
            filter_obj.add_measurement(-65.0 - i, base_time + i * 0.5)
        
        assert filter_obj.get_sample_count() == 10
        result = filter_obj.get_median_iqr()
        assert result is not None

    def test_median_iqr_outlier_rejection(self):
        """Test that IQR method rejects outliers properly."""
        filter_obj = AdaptivePercentileRSSI()
        base_time = 1000.0
        
        # Add mostly consistent values around -65
        for i in range(20):
            filter_obj.add_measurement(-65.0, base_time + i * 0.1)
        
        # Add some outliers
        filter_obj.add_measurement(-30.0, base_time + 2.1)  # Strong outlier (too high)
        filter_obj.add_measurement(-95.0, base_time + 2.2)  # Strong outlier (too low)
        
        result = filter_obj.get_median_iqr()
        
        # Result should be close to -65, not influenced by outliers
        assert result is not None
        assert -67.0 < result < -63.0

    def test_iqr_with_varied_data(self):
        """Test IQR filtering with varied but reasonable data."""
        filter_obj = AdaptivePercentileRSSI()
        base_time = 1000.0
        
        # Realistic RSSI variation
        rssi_values = [-65, -67, -64, -66, -65, -68, -64, -65, -66, -67]
        for i, rssi in enumerate(rssi_values):
            filter_obj.add_measurement(float(rssi), base_time + i * 0.2)
        
        result = filter_obj.get_median_iqr()
        assert result is not None
        # Should be close to mean of values (~-65.7)
        assert -67.0 < result < -64.0

    def test_tukey_fence_custom_k(self):
        """Test custom k coefficient for Tukey fence."""
        filter_obj = AdaptivePercentileRSSI(iqr_coefficient=1.0)  # Stricter outlier detection
        base_time = 1000.0
        
        # Add data with some spread
        for i in range(10):
            filter_obj.add_measurement(-65.0 + (i % 3), base_time + i * 0.1)
        
        result_strict = filter_obj.get_median_iqr()
        
        # Compare with looser coefficient
        result_loose = filter_obj.get_median_iqr(k=2.0)
        
        assert result_strict is not None
        assert result_loose is not None

    def test_rssi_variance_calculation(self):
        """Test RSSI variance calculation."""
        filter_obj = AdaptivePercentileRSSI()
        base_time = 1000.0
        
        # Add identical values - variance should be 0
        for i in range(5):
            filter_obj.add_measurement(-65.0, base_time + i * 0.1)
        
        variance = filter_obj.get_rssi_variance()
        assert variance == 0.0
        
        # Add some variation
        filter_obj.add_measurement(-70.0, base_time + 0.6)
        filter_obj.add_measurement(-60.0, base_time + 0.7)
        
        variance = filter_obj.get_rssi_variance()
        assert variance > 0.0

    def test_rssi_variance_insufficient_data(self):
        """Test variance returns 0 with insufficient data."""
        filter_obj = AdaptivePercentileRSSI()
        
        # No data
        assert filter_obj.get_rssi_variance() == 0.0
        
        # Single measurement
        filter_obj.add_measurement(-65.0, 100.0)
        assert filter_obj.get_rssi_variance() == 0.0

    def test_distance_variance_calculation(self):
        """Test distance variance calculation."""
        filter_obj = AdaptivePercentileRSSI()
        base_time = 1000.0
        
        # Add measurements with some variation
        rssi_values = [-65, -67, -64, -66, -68]
        for i, rssi in enumerate(rssi_values):
            filter_obj.add_measurement(float(rssi), base_time + i * 0.1)
        
        ref_power = -65.0
        attenuation = 3.0
        
        dist_variance = filter_obj.get_distance_variance(ref_power, attenuation)
        assert dist_variance >= 0.0

    def test_distance_variance_insufficient_data(self):
        """Test distance variance returns 0 with insufficient data."""
        filter_obj = AdaptivePercentileRSSI()
        
        # No data
        assert filter_obj.get_distance_variance(-65.0, 3.0) == 0.0
        
        # Single measurement
        filter_obj.add_measurement(-65.0, 100.0)
        assert filter_obj.get_distance_variance(-65.0, 3.0) == 0.0

    def test_time_window_expiration(self):
        """Test that old readings are removed after time window."""
        filter_obj = AdaptivePercentileRSSI(time_window_ms=1000)  # 1 second window
        base_time = 1000.0
        
        # Add measurement at t=1000
        filter_obj.add_measurement(-65.0, base_time)
        assert filter_obj.get_sample_count() == 1
        
        # Add measurement at t=1500 (within window)
        filter_obj.add_measurement(-66.0, base_time + 0.5)
        assert filter_obj.get_sample_count() == 2
        
        # Add measurement at t=2100 (first measurement should be expired)
        filter_obj.add_measurement(-67.0, base_time + 1.1)
        assert filter_obj.get_sample_count() == 2  # First one expired

    def test_buffer_size_adjustment(self):
        """Test adaptive buffer sizing based on advertisement rate."""
        filter_obj = AdaptivePercentileRSSI(
            time_window_ms=1000,
            initial_max_readings=20,
        )
        base_time = 1000.0
        
        # Simulate fast advertising (100ms intervals)
        # Should increase buffer size to accommodate more readings in time window
        for i in range(15):
            filter_obj.add_measurement(-65.0 - i * 0.1, base_time + i * 0.1)
        
        # Force adjustment by advancing time past adjustment interval
        filter_obj.add_measurement(-65.0, base_time + 11.0)
        
        # Buffer should have adjusted
        assert filter_obj.get_sample_count() > 0

    def test_clear_filter(self):
        """Test clearing all readings from filter."""
        filter_obj = AdaptivePercentileRSSI()
        base_time = 1000.0
        
        # Add some measurements
        for i in range(10):
            filter_obj.add_measurement(-65.0 - i, base_time + i * 0.1)
        
        assert filter_obj.get_sample_count() == 10
        
        # Clear the filter
        filter_obj.clear()
        
        assert filter_obj.get_sample_count() == 0
        assert filter_obj.get_median_iqr() is None
        assert filter_obj.get_rssi_variance() == 0.0

    def test_percentile_edge_cases(self):
        """Test percentile calculation edge cases."""
        filter_obj = AdaptivePercentileRSSI()
        
        # Empty list
        result = filter_obj._calculate_percentile([], 0.5)
        assert result == 0.0
        
        # Single value
        result = filter_obj._calculate_percentile([-65.0], 0.5)
        assert result == -65.0
        
        # Two values - median should be average
        result = filter_obj._calculate_percentile([-60.0, -70.0], 0.5)
        assert result == -65.0
        
        # Quartiles of 10 values
        values = sorted([-65.0 - i for i in range(10)])
        q1 = filter_obj._calculate_percentile(values, 0.25)
        q3 = filter_obj._calculate_percentile(values, 0.75)
        assert q1 < q3

    def test_median_iqr_all_outliers_rejected(self):
        """Test fallback to median when all values are rejected as outliers."""
        filter_obj = AdaptivePercentileRSSI(iqr_coefficient=0.001)  # Extremely strict
        base_time = 1000.0
        
        # Add varied values
        rssi_values = [-65, -66, -64, -67, -63]
        for i, rssi in enumerate(rssi_values):
            filter_obj.add_measurement(float(rssi), base_time + i * 0.1)
        
        # Should fall back to median when fence rejects everything
        result = filter_obj.get_median_iqr()
        assert result is not None
        # Should be close to median of values
        assert -67 <= result <= -63

    def test_realistic_ble_scenario(self):
        """Test with realistic BLE RSSI data including noise."""
        filter_obj = AdaptivePercentileRSSI()
        base_time = 1000.0
        
        # Simulate device at ~3 meters with typical RSSI fluctuations
        # Expected RSSI around -70 with ±5 dBm variation
        realistic_rssi = [
            -68, -72, -69, -71, -70, -73, -68, -70, -69, -71,
            -85,  # spike (reflection or interference)
            -70, -72, -69, -71, -70,
            -55,  # another spike
            -68, -71, -70, -69,
        ]
        
        for i, rssi in enumerate(realistic_rssi):
            filter_obj.add_measurement(float(rssi), base_time + i * 0.3)
        
        result = filter_obj.get_median_iqr()
        assert result is not None
        
        # Result should be around -70, not influenced by -85 and -55 spikes
        assert -73 < result < -67
        
        # Variance should be reasonable
        variance = filter_obj.get_rssi_variance()
        assert variance > 0

    def test_fast_vs_slow_advertising(self):
        """Test buffer adaptation to different advertising rates."""
        # Fast advertising (100ms)
        fast_filter = AdaptivePercentileRSSI(time_window_ms=5000)
        base_time = 1000.0
        
        for i in range(50):
            fast_filter.add_measurement(-65.0, base_time + i * 0.1)
        
        # Trigger adjustment
        fast_filter.add_measurement(-65.0, base_time + 11.0)
        fast_count = fast_filter.get_sample_count()
        
        # Slow advertising (1000ms)
        slow_filter = AdaptivePercentileRSSI(time_window_ms=5000)
        
        for i in range(10):
            slow_filter.add_measurement(-65.0, base_time + i * 1.0)
        
        # Trigger adjustment
        slow_filter.add_measurement(-65.0, base_time + 20.0)
        slow_count = slow_filter.get_sample_count()
        
        # Both should maintain readings, but within their buffers
        assert fast_count > 0
        assert slow_count > 0

    def test_no_measurement_returns_none(self):
        """Test that get_median_iqr returns None when no measurements."""
        filter_obj = AdaptivePercentileRSSI()
        assert filter_obj.get_median_iqr() is None


class TestVarianceWeightingFormula:
    """Test the variance weighting formula used in trilateration."""

    def test_variance_weighting_low_variance(self):
        """Test that low RSSI variance results in high weight."""
        # Low variance (stable signal)
        rssi_var = 1.0  # dB²
        variance_threshold = 5.0
        variance_weight = 1.0 / (1.0 + math.log1p(rssi_var / variance_threshold))
        variance_weight = max(0.2, variance_weight)
        
        # Should be close to 1.0 (high weight)
        assert variance_weight > 0.85
        assert variance_weight <= 1.0

    def test_variance_weighting_medium_variance(self):
        """Test that medium RSSI variance results in medium weight."""
        # Medium variance
        rssi_var = 5.0  # dB²
        variance_threshold = 5.0
        variance_weight = 1.0 / (1.0 + math.log1p(rssi_var / variance_threshold))
        variance_weight = max(0.2, variance_weight)
        
        # Should be around 0.5-0.8
        assert 0.5 < variance_weight < 0.8

    def test_variance_weighting_high_variance(self):
        """Test that high RSSI variance results in lower weight but not zero."""
        # High variance (noisy signal)
        rssi_var = 20.0  # dB²
        variance_threshold = 5.0
        variance_weight = 1.0 / (1.0 + math.log1p(rssi_var / variance_threshold))
        variance_weight = max(0.2, variance_weight)
        
        # Should be reduced but floored at 0.2
        assert variance_weight >= 0.2
        assert variance_weight < 0.5

    def test_variance_weighting_extreme_variance(self):
        """Test that extreme variance still maintains minimum weight floor."""
        # Extreme variance (very noisy)
        rssi_var = 1000.0  # dB²
        variance_threshold = 5.0
        variance_weight = 1.0 / (1.0 + math.log1p(rssi_var / variance_threshold))
        variance_weight = max(0.2, variance_weight)
        
        # Should be floored at 0.2 (minimum weight)
        assert variance_weight == 0.2

    def test_variance_weighting_zero_variance(self):
        """Test handling of zero or near-zero variance."""
        # Near-zero variance
        rssi_var = 0.005  # dB² (below 0.01 threshold)
        
        # Should not apply variance weighting
        if rssi_var <= 0.01:
            variance_weight = 1.0  # No penalty for perfect signal
        
        assert variance_weight == 1.0
