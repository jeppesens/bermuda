# Quick Start: Trilateration in 5 Minutes

Get Bermuda trilateration up and running quickly with this condensed guide.

## Prerequisites
- Bermuda BLE Trilateration integration installed
- 2+ bluetooth proxies (ESPHome/Shelly) already tracking devices
- Ability to measure distances in meters

## Steps

### 1. Find Your Scanner MAC Addresses (2 minutes)

**Option A:** Via Bermuda dump service
1. Developer Tools → Services → `bermuda.dump_devices`
2. Call service
3. Look for entries with `"is_scanner": true`
4. Copy the `"address"` field (e.g., `aa:bb:cc:dd:ee:ff`)

**Option B:** Via Devices UI
1. Settings → Devices & Services → Bermuda BLE Trilateration
2. Click each proxy device
3. Copy MAC address from device info

### 2. Import Positions via UI (2 minutes)

1. Settings → Devices & Services → Bermuda BLE Trilateration
2. Click **⋮ menu** (three dots) → **Bulk Import Map & Scanners**
3. Paste this JSON (replace MAC addresses with your scanners):

```json
{
  "nodes": [
    {
      "id": "PUT_YOUR_SCANNER_MAC_HERE",
      "name": "Living Room",
      "point": [0.0, 0.0, 1.0]
    },
    {
      "id": "PUT_YOUR_SCANNER_MAC_HERE",
      "name": "Bedroom",
      "point": [5.0, 0.0, 1.0]
    },
    {
      "id": "PUT_YOUR_SCANNER_MAC_HERE",
      "name": "Kitchen",
      "point": [2.5, 4.0, 1.0]
    }
  ]
}
```

4. Click **"Replace All"** → **Submit**

**Quick positioning tips:**
- Don't stress about precision initially - you can refine later
- Just estimate room corners in meters from an arbitrary origin
- Set z (height) to 1.0 for wall-mounted, 0.5 for low placement
- Use graph paper or measure with tape if available

### 3. Reload Integration (10 seconds)

Settings → Devices & Services → Bermuda → **⋮ menu** → **Reload**

(Or restart Home Assistant if you prefer)

### 4. Enable Position Sensor (30 seconds)

1. Go to your tracked device (e.g., phone, beacon)
2. Find sensor ending in "Position" (disabled by default)
3. Click → Settings (gear) → Enable

### 5. Verify It Works!

Check the Position sensor:
- State shows coordinates: `(2.5, 3.1, 1.0)`
- Attributes show confidence and scanner_count

**Not working?** Check logs:
```
Settings → System → Logs → Filter: "bermuda"
```

Look for:
- `Loaded X scanner positions for trilateration`
- Any errors about JSON parsing or scanner MAC mismatches

## Next Steps

Once basic positioning works:

1. **Refine coordinates** - Measure actual positions for accuracy
2. **Add more scanners** - 4+ gives best results
3. **Calibrate RSSI** - Adjust Bermuda's `ref_power` and `attenuation` settings
4. **Use in automations** - Filter by `confidence` attribute

See [TRILATERATION.md](TRILATERATION.md) for full documentation.

## Common Issues

**Sensor stays "unknown"**
- Check MAC addresses match exactly (case-insensitive but format matters)
- Ensure JSON is valid (use JSONLint.com to validate)
- Verify positions were imported successfully (check logs for "Loaded X scanner positions")
- Try re-importing via Bulk Import menu

**Low confidence (<50%)**
- Add more scanners
- Check distance sensors show reasonable values (not all >20m)
- Verify scanner coordinates are approximately correct

**Position jumps around**
- Increase smoothing_samples in Bermuda config
- Add more scanners for better coverage
- Reduce max_velocity to reject large jumps

Need help? [Ask in the community forum](https://community.home-assistant.io/t/bermuda-bluetooth-ble-room-presence-and-tracking-custom-integration/625780/1)
