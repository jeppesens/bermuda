# Bermuda BLE Trilateration

## What This Is

A Home Assistant custom integration (fork of [agittins/bermuda](https://github.com/agittins/bermuda)) that tracks BLE device positions indoors using ESP32-based BLE proxy scanners. This fork (by jeppesens) adds ESPresense-style position tracking with coordinate-based trilateration.

## Project Goal

Match the position tracking quality of [ESPresense](https://espresense.com/) companion, but implemented entirely within the Home Assistant integration (no separate service). Uses the same algorithms and configuration format as ESPresense.

## Architecture

```
custom_components/bermuda/
  __init__.py          # HA integration setup, BermudaData/BermudaConfigEntry
  coordinator.py       # Main data update loop, YAML config loading, position pipeline
  bermuda_device.py    # BermudaDevice model (tracked devices + scanners)
  bermuda_advert.py    # BermudaAdvert - RSSI advertisements from scanners
  trilateration.py     # Position algorithms: Nadaraya-Watson + nearest_node
  kalman.py            # 6-state Kalman filter (position + velocity smoothing)
  config_flow.py       # HA config/options UI, bulk import (JSON + ESPresense YAML)
  sensor.py            # HA sensor entities (distance, position, RSSI, etc.)
  const.py             # All constants, config keys, defaults
  util.py              # Helpers: MAC normalization, RSSI-to-distance, scanner validation
  device_tracker.py    # HA device tracker entity
  binary_sensor.py     # Occupancy binary sensors
  number.py / switch.py # Config number/switch entities
```

## Key Algorithms

- **Nadaraya-Watson kernel regression** (`trilateration.py`): Inverse-distance-squared weighting for position estimation. Used with 2+ scanners.
- **Nearest Node** (`trilateration.py`): Fallback for 1 scanner - uses scanner's position directly.
- **6-state Kalman filter** (`kalman.py`): Tracks `[x, y, z, vx, vy, vz]` with constant-velocity model. Ported from ESPresense's `KalmanLocation.cs`. Pure Python, no numpy.
- **RSSI to distance**: Friis path-loss model in `util.py`.
- **Point-in-polygon**: Ray casting for room detection in `trilateration.py`.

## Configuration

Supports two config formats:
1. **ESPresense YAML** (`bermuda.yaml` in HA config dir): Floors with nested rooms, nodes with `point`/`floors`, filtering settings. Auto-loaded by coordinator.
2. **JSON bulk import**: Via config flow UI. Flat format with explicit floor references.

## Key Patterns

- `coordinator.py` is the heart - `_async_update_data()` runs the full pipeline every cycle
- Scanner positions come from config (YAML or JSON import), stored in `device.position`
- Each tracked device has per-device `_kalman_location` (lazily initialized)
- Position pipeline: gather adverts -> validate scanners -> filter by floor -> Nadaraya-Watson/nearest_node -> Kalman filter -> find room -> update device
- All position data stored on `BermudaDevice`: `calculated_position`, `position_confidence`, `position_method`, `position_error`, `position_correlation`, `position_room_id`, `position_floor_id`

## Development

- Python 3.12+ required (uses `type` statement in `__init__.py`)
- No external dependencies beyond HA core (`requirements: []` in manifest)
- Tests in `tests/` - use `pytest` with `pytest-homeassistant-custom-component`
- Kalman filter and trilateration modules can be tested in isolation (no HA dependency)
- The `.venv` uses Python 3.10 which can't run the full test suite due to `type` syntax; use pyenv 3.12+ for that

## Branch Info

- `trilateration` branch: Active development branch with ESPresense-style positioning
- `main` branch: Upstream sync target
- Forked from `agittins/bermuda` - upstream has NO trilateration code
