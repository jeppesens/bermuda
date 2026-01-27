# RSSI Filtering and Variance-Based Trilateration

## Overview

Bermuda now includes advanced RSSI filtering inspired by ESPresence's implementation, using adaptive percentile-based outlier rejection with IQR (Interquartile Range) method. This significantly improves distance estimation accuracy in noisy BLE environments.

## Features

### 1. Adaptive Percentile RSSI Filtering

The `AdaptivePercentileRSSI` class provides robust outlier rejection using **Tukey's outlier detection method**:

- **Time-windowed buffer**: Maintains a 15-second rolling window of RSSI readings
- **Adaptive buffer sizing**: Automatically adjusts buffer size (10-200 samples) based on device advertisement rate
- **IQR outlier rejection**: Uses Tukey fence `[Q1 - k*IQR, Q3 + k*IQR]` to statistically reject outlier RSSI spikes
- **Variance tracking**: Calculates both RSSI variance (dB²) and distance variance (m²)

#### How It Works

```
BLE Advertisement (RSSI=-70)
  ↓
AdaptivePercentileRSSI.add_measurement()
  ↓
Time-windowed buffer (15s, auto-sized)
  ↓
Calculate Q1, Median, Q3 → IQR
  ↓
Apply Tukey fence [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
  ↓
Mean of survivors → Filtered RSSI
  ↓
Calculate RSSI & distance variance
  ↓
Trilateration uses variance for weighting
```

### 2. Variance-Based Trilateration Weighting

Scanners with higher measurement variance (less stable signals) receive lower weight in trilateration calculations:

- **RSSI variance** (not distance variance) is used for stability
- **Logarithmic weighting formula**: `variance_weight = 1.0 / (1.0 + log(1 + rssi_var/5.0))`
- **Minimum weight floor**: Even noisy scanners contribute at least 20% weight
- **Prevents weight collapse**: Ensures `total_weight > 0` always

## Configuration

### RSSI Filter Mode

```yaml
# Configuration options (set via UI or configuration.yaml)
rssi_filter_mode: "median_iqr"  # or "legacy"
```

**Options:**
- `median_iqr` (default): Adaptive percentile-based filtering with outlier rejection
- `legacy`: Simple rolling average (backward compatibility)

### IQR Coefficient

```yaml
iqr_coefficient: 1.5  # Range: 1.0-2.0
```

Controls strictness of outlier detection:
- **1.0**: Stricter outlier rejection (more aggressive filtering)
- **1.5**: Standard Tukey fence (default, balanced)
- **2.0**: Looser outlier rejection (less aggressive, keeps more data)

### RSSI Time Window

```yaml
rssi_time_window_ms: 15000  # milliseconds
```

Time window for RSSI filtering (default: 15000ms = 15 seconds)
- Longer windows: More stable but slower to respond to movement
- Shorter windows: Faster response but more noise

### Variance Weighting

```yaml
trilateration_use_variance_weighting: true  # or false
```

Enable/disable variance-based weighting in trilateration:
- `true` (default): Use RSSI variance to weight scanner reliability
- `false`: Disable variance weighting (all scanners weighted equally by distance/index only)

## Performance Characteristics

### Variance Weighting Formula

| RSSI Variance (dB²) | Variance Weight | Scanner Contribution |
|---------------------|-----------------|----------------------|
| 0-2 (excellent)     | 0.90-1.00       | High priority        |
| 2-5 (good)          | 0.60-0.90       | Medium-high priority |
| 5-10 (acceptable)   | 0.40-0.60       | Medium priority      |
| 10-20 (noisy)       | 0.25-0.40       | Low priority         |
| 20+ (very noisy)    | 0.20 (floor)    | Minimum contribution |

### Expected RSSI Variance

- **Stationary device, good signal**: 1-3 dB²
- **Stationary device, normal**: 3-8 dB²
- **Moving device**: 5-15 dB²
- **Obstructed/reflective environment**: 10-25 dB²
- **Poor signal quality**: 20+ dB²

## Comparison: Legacy vs Median IQR

| Aspect | Legacy | Median IQR |
|--------|--------|------------|
| Outlier Rejection | Velocity-based | Statistical (Tukey IQR) |
| Buffer Management | Fixed size | Adaptive to ad rate |
| Noise Immunity | Basic smoothing | Robust percentile filtering |
| Quality Metrics | None | Variance tracking |
| Confidence Scoring | Distance only | Distance + variance |
| BLE spike handling | Poor | Excellent |
| Fast-moving devices | Good | Good |
| Multi-floor environments | Fair | Excellent |

## Troubleshooting

### High Variance Values

If you see consistently high RSSI variance (>20 dB²) in logs:

1. **Check scanner placement**: Avoid mounting near metal surfaces or Wi-Fi routers
2. **Verify power supply**: Unstable power can cause erratic RSSI readings
3. **Update ESPHome**: Older versions may have less stable BLE scanning
4. **Calibrate ref_power**: Incorrect calibration amplifies variance effects
5. **Increase time window**: Try `rssi_time_window_ms: 20000` for more stability

### Trilateration Weight Collapse

If you see warnings like `total_weight=0.0000`:

1. **Automatic fallback**: System falls back to unweighted centroid (all scanners equal)
2. **Check variance weighting**: Try `trilateration_use_variance_weighting: false` temporarily
3. **Verify scanner health**: Use `bermuda.dump_devices` to check scanner states
4. **Review logs**: Look for `rssi_variance` values in debug logs

### Slow Response to Movement

If position updates lag behind actual movement:

1. **Reduce time window**: Try `rssi_time_window_ms: 10000` (10 seconds)
2. **Adjust IQR coefficient**: Try `iqr_coefficient: 2.0` (less aggressive filtering)
3. **Check update interval**: Ensure `update_interval` is not too long (default: 10s)

## Debug Logging

Enable trilateration debug logging to see variance metrics:

```yaml
trilateration_debug: true
```

Look for log entries like:
```
Scanner 0: rssi_variance=5.3 dB², variance_weight=0.591
Scanner 1: rssi_variance=12.7 dB², variance_weight=0.337
```

## Best Practices

1. **Start with defaults**: The default settings work well for most environments
2. **Calibrate scanners properly**: Accurate `ref_power` is critical for variance calculations
3. **Monitor variance**: Check logs periodically to ensure variance values are reasonable
4. **Use multiple scanners**: Variance weighting works best with 3+ scanners per area
5. **Test before deploying**: Validate in your environment before relying on it for automations

## Technical Details

### Tukey's Outlier Detection

The IQR method identifies outliers using the Tukey fence:

1. Calculate quartiles: Q1 (25th percentile), Median (50th), Q3 (75th percentile)
2. Calculate IQR = Q3 - Q1
3. Define fence: `[Q1 - k*IQR, Q3 + k*IQR]` where k=1.5 (default)
4. Reject values outside fence as outliers
5. Return mean of remaining values (or median if all rejected)

### Variance Calculation

**RSSI Variance** (in dB²):
```python
variance = sum((rssi - mean)² for rssi in readings) / len(readings)
```

**Distance Variance** (in m²):
```python
# Convert each RSSI to distance, then calculate variance
distances = [10^((ref_power - rssi) / (10 * attenuation)) for rssi in readings]
variance = sum((d - mean)² for d in distances) / len(distances)
```

**Note**: Trilateration uses RSSI variance (not distance variance) because:
- RSSI variance is stable: typically 1-20 dB²
- Distance variance is exponentially amplified: can reach 1000s of m²
- RSSI variance better represents signal quality

### Adaptive Buffer Sizing

Buffer size adjusts based on advertisement rate:

```python
# Fast advertising (100ms intervals) → larger buffer (100+ samples)
# Slow advertising (1000ms intervals) → smaller buffer (15-20 samples)
target_size = time_window_ms / (avg_interval_ms)
target_size = max(10, min(target_size, 200))
```

This ensures consistent time-window coverage regardless of device advertisement rate.

## Future Enhancements

Potential improvements under consideration:

- **Kalman filtering**: Predictive filtering for moving devices
- **Environment-adaptive thresholds**: Auto-tune based on observed variance patterns
- **Per-scanner variance tracking**: Historical variance profiles for each scanner
- **Variance-based confidence scoring**: Expose variance metrics to users for debugging
- **Machine learning**: Anomaly detection for unusual RSSI patterns

## References

- [ESPresense Implementation](https://github.com/ESPresense/ESPresense/blob/main/lib/filtering/AdaptivePercentileRSSI.cpp)
- [Tukey's Outlier Detection](https://en.wikipedia.org/wiki/Outlier#Tukey's_fences)
- [IQR Method](https://en.wikipedia.org/wiki/Interquartile_range)
- [BLE RSSI Characteristics](https://www.bluetooth.com/blog/bluetooth-low-energy-rssi/)
