# Trilateration Setup Guide

This guide explains how to set up and use Bermuda's trilateration feature to calculate precise 3D positions of your tracked bluetooth devices.

## Overview

Trilateration calculates a device's (x, y, z) position in 3D space by measuring distances from multiple bluetooth proxies with known coordinates. This is similar to GPS, but works indoors using bluetooth signal strength (RSSI) to estimate distances.

## Requirements

- **Minimum 2 bluetooth proxies** (scanners) for basic positioning
- **3+ proxies recommended** for reliable 2D positioning
- **4+ proxies recommended** for accurate 3D positioning
- Each proxy must have its physical position configured in meters

## Scanner Count vs Accuracy

| Scanners | Capability | Confidence | Use Case |
|----------|-----------|------------|----------|
| 1 | Distance only | Very Low (10-20%) | Fallback when device only visible to one proxy |
| 2 | 2D with ambiguity | Medium (50-60%) | Small rooms, hallways |
| 3 | Reliable 2D, potential 3D | Good (65-80%) | Most rooms, general tracking |
| 4+ | Full 3D positioning | High (80-95%) | Precise location, multi-floor homes |

## Setup Instructions

### Step 1: Measure Your Space

You need to assign (x, y, z) coordinates to each bluetooth proxy. Choose a coordinate system:

1. **Pick an origin point** - typically the bottom-left corner of your floor plan at ground level
2. **Measure in meters** - all coordinates must be in meters
3. **Define axes**:
   - **x-axis**: Left to right (west to east)
   - **y-axis**: Bottom to top (south to north)  
   - **z-axis**: Height above ground/floor

**Example for a 10m x 8m room:**
```
(0, 8, 0)  -------- (10, 8, 0)
    |                    |
    |                    |
    |       Room         |
    |                    |
    |                    |
(0, 0, 0)  -------- (10, 0, 0)
```

### Step 2: Find Scanner MAC Addresses

1. Go to **Settings** → **Devices & Services** → **Bermuda BLE Trilateration**
2. Click on the **Devices** tab
3. Find your bluetooth proxies (ESPHome devices, Shelly devices)
4. Note the MAC address for each scanner (shown in device details)

Alternatively, use the `bermuda.dump_devices` service and look for devices with `is_scanner: true`.

### Step 3: Import Scanner Positions via UI

Navigate to the Bermuda integration's bulk import feature:

1. Go to **Settings** → **Devices & Services** → **Bermuda BLE Trilateration**
2. Click the **⋮ menu** (three dots in top right)
3. Select **"Bulk Import Map & Scanners"**
4. **Paste your JSON configuration** into the text field

**Format:**
```json
{
  "nodes": [
    {
      "id": "AA:BB:CC:DD:EE:FF",
      "name": "Living Room Proxy",
      "point": [0.5, 0.5, 1.2]
    },
    {
      "id": "11:22:33:44:55:66",
      "name": "Bedroom Proxy",
      "point": [5.0, 0.5, 1.2]
    },
    {
      "id": "99:88:77:66:55:44",
      "name": "Kitchen Proxy",
      "point": [2.5, 4.5, 1.2]
    }
  ]
}
```

**Field Definitions:**
- `id`: **Required** - MAC address of the scanner (lowercase, colon-separated)
- `name`: **Optional** - Friendly name for logging/debugging
- `point`: **Required** - Array of 3 numbers `[x, y, z]` in meters

**Tips:**
- MAC addresses must match exactly (case-insensitive, but format matters)
- Use `00:00:00:00:00:00` format (colons, not dashes or underscores)
- All three coordinates (x, y, z) are always required
- Typical proxy heights: 0.5m-2.0m above floor
- See `scanner_positions.json.example` for a complete format example

5. Choose **"Replace All"** (removes existing positions) or **"Merge"** (adds to existing)
6. Click **Submit**

### Step 4: Reload the Integration

After importing scanner positions, reload the integration to apply changes:

1. Go to **Settings** → **Devices & Services** → **Bermuda BLE Trilateration**
2. Click the **⋮ menu** → **Reload**

Alternatively, restart Home Assistant.

Check the logs for confirmation:
```
INFO (MainThread) [custom_components.bermuda.coordinator] Loaded position for scanner Living Room Proxy (aa:bb:cc:dd:ee:ff): (0.50, 0.50, 1.20)
INFO (MainThread) [custom_components.bermuda.coordinator] Loaded 3 scanner positions for trilateration
```

### Step 5: Enable Position Sensors

1. Go to your tracked device in Home Assistant
2. Find the **Position** sensor (disabled by default)
3. Click the entity, go to Settings (gear icon)
4. Enable the entity

The sensor will show coordinates like `(2.5, 3.1, 1.0)` with attributes:
- `x`, `y`, `z`: Individual coordinates
- `confidence`: 0-100% accuracy estimate
- `method`: Algorithm used (`2-scanner`, `3-scanner`, `4+scanner`)
- `scanner_count`: Number of scanners used

## Algorithm Details

### Weighted Centroid (3+ scanners)

For 3 or more scanners, Bermuda uses a weighted centroid algorithm:

```
weight = 1 / (distance² + ε)
position = Σ(scanner_position × weight) / Σ(weight)
```

This gives higher influence to closer scanners (which have more accurate distance measurements).

### Bilateration (2 scanners)

With 2 scanners, there are two possible intersection points. Bermuda chooses between them using:
1. **Previous position** - picks point closer to last known location
2. **Velocity limits** - rejects movements faster than `max_velocity` (default 3 m/s)
3. **Fallback** - uses weighted centroid if intersection fails

### Distance-Only (1 scanner)

With 1 scanner, only distance is known, not direction. Bermuda:
1. If previous position exists, maintains direction and scales to new distance
2. Otherwise, reports scanner's position with very low confidence

## Configuration Options

Trilateration respects these Bermuda settings:

### Enable/Disable Trilateration
Set in `configuration.yaml` (advanced users):
```yaml
bermuda:
  enable_trilateration: true  # Default: true
  min_trilateration_scanners: 2  # Default: 2 (minimum 1, maximum 5)
```

### Related Settings

- `max_velocity` (default 3 m/s): Reject position changes implying faster movement
- `max_area_radius` (default 20m): Ignore scanners beyond this distance
- `smoothing_samples` (default 20): More samples = smoother distance measurements

## Troubleshooting

### Position sensor shows "unknown" or "unavailable"

**Possible causes:**
1. Scanner positions not imported or invalid JSON format
2. Scanner MAC addresses don't match (check capitalization, format)
3. Not enough scanners with valid distances (check distance sensors)
4. Trilateration disabled in configuration

**Solutions:**
- Check HA logs for "Loaded X scanner positions" message
- Verify MAC addresses match exactly using `bermuda.dump_devices`
- Re-import scanner positions via **Bulk Import Map & Scanners** menu
- Ensure scanners have valid distance measurements to the device
- Enable debug logging: `custom_components.bermuda: debug`

### Positions seem inaccurate

**Common issues:**
1. **Incorrect scanner coordinates** - re-measure and verify
2. **Poor scanner placement** - add more scanners or reposition for better coverage
3. **RSSI calibration needed** - adjust `ref_power` and `attenuation` in Bermuda settings
4. **Environmental factors** - walls, metal objects, interference affect signals

**Optimization tips:**
- Place scanners at room corners for best geometry
- Vary scanner heights (z-coordinate) for better 3D accuracy
- Use 4+ scanners for critical areas
- Calibrate RSSI settings per-scanner for best distance accuracy

### Position jumps or jitters

This usually indicates poor signal or scanner coverage. Solutions:

1. Increase `smoothing_samples` for more stable distance measurements
2. Add more scanners to improve confidence
3. Check `confidence` attribute - filter out low-confidence positions in automations
4. Reduce `max_velocity` to prevent large jumps (but not too low)

### Confidence always low

Low confidence (< 50%) can indicate:
- Only 1-2 scanners seeing the device
- Large variance in distance measurements (signal interference)
- Scanner positions may be incorrectly configured
- RSSI calibration needed

## Example Automation

Filter unreliable positions using confidence threshold:

```yaml
automation:
  - alias: "Track phone when confidence high"
    trigger:
      - platform: state
        entity_id: sensor.my_phone_position
    condition:
      - condition: template
        value_template: "{{ state_attr('sensor.my_phone_position', 'confidence') | float > 60 }}"
    action:
      - service: notify.mobile_app
        data:
          message: "Phone at ({{ state_attr('sensor.my_phone_position', 'x') }}, {{ state_attr('sensor.my_phone_position', 'y') }})"
```

## ESPresense Compatibility

Bermuda's `scanner_positions.json` format is inspired by ESPresense but simplified. 

**ESPresense users:** You can extract node positions from your ESPresense configuration, but note:
- Bermuda only uses the `nodes` array (rooms, floors ignored for now)
- `point` must be 3 values `[x, y, z]` - 2D points not supported
- GPS coordinates, rotation, bounds are not used (future feature)

## Advanced: Creating Floor Plans

While Bermuda doesn't currently visualize positions on a map, you can:

1. **Export positions** using `bermuda.dump_devices` service
2. **Create template sensors** that combine x, y coordinates
3. **Use map card** (custom card) to plot positions on floor plan image
4. **Integrate with node-red** for custom visualizations

See [the Wiki](https://github.com/agittins/bermuda/wiki/) for community examples.

## Theory: How Trilateration Works

Trilateration determines position by measuring distances to known reference points:

1. **Distance estimation**: RSSI (signal strength) → distance using path loss formula
2. **Sphere intersection**: Each scanner defines a sphere of radius = distance
3. **Solve for position**: Find point where all spheres intersect
4. **Weighted averaging**: Give more weight to closer (more accurate) measurements

**Why it's harder than GPS:**
- Bluetooth signals are affected by walls, furniture, people
- RSSI-to-distance conversion is approximate
- Indoor multipath (signal bouncing) creates noise
- Scanners may not have perfectly synchronized clocks

That's why Bermuda uses statistical smoothing, velocity limits, and confidence scoring to produce reliable results despite these challenges.

## Future Enhancements

Planned features:
- Visual floor plan with position overlay
- Room/zone detection using polygon containment
- Multi-floor support with automatic floor detection
- GPS coordinate conversion for outdoor tracking
- Automatic scanner position calibration (fingerprinting)
- Kalman filtering for smoother position tracking
- Path prediction and velocity estimation

## Support

- [Bermuda Wiki](https://github.com/agittins/bermuda/wiki/)
- [GitHub Discussions](https://github.com/agittins/bermuda/discussions/)
- [Home Assistant Community Forum](https://community.home-assistant.io/t/bermuda-bluetooth-ble-room-presence-and-tracking-custom-integration/625780/1)
