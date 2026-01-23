# Bermuda BLE Trilateration - AI Coding Agent Instructions

## Project Overview
Bermuda is a Home Assistant custom integration that performs BLE-based room presence detection and trilateration using ESPHome bluetooth proxies and Shelly devices. It tracks bluetooth devices by calculating distances from RSSI values and determining which Area (room) they're in.

## Architecture & Core Concepts

### Device Hierarchy (Critical)
Bermuda has a unique device model - understand these distinctions:

1. **BermudaDevice** (`bermuda_device.py`): Internal representation, NOT HA entities. Every discovered BT transmitter/receiver gets one, even if not tracked. Stored in `coordinator.devices[address]` dict.
2. **Scanners**: BT receivers (ESPHome proxies, Shelly devices) that detect advertisements
3. **Tracked Devices**: Subset of BermudaDevices where `create_sensor=True`, which get sensor entities
4. **Meta-devices**: Virtual devices that aggregate multiple MAC addresses:
   - iBeacon meta-devices (tracked by UUID) aggregate source MACs that broadcast the beacon
   - Private BLE devices (IRK-based) aggregate rotating random MAC addresses
   - Check `METADEVICE_TYPE_*` and `METADEVICE_*_DEVICE` constants in `const.py`

### Data Flow
```
BT Advertisement → HA Bluetooth Integration → BermudaDataUpdateCoordinator
  ↓ (UPDATE_INTERVAL = 1.05s processing)
  ├─ Process advertisements (BermudaAdvert objects)
  ├─ Calculate distances using rssi_to_metres()
  ├─ Determine closest scanner per device
  └─ Update area assignments
       ↓ (CONF_UPDATE_INTERVAL throttling, default 10s)
       Sensor/DeviceTracker entities update states
```

### Key Files & Responsibilities
- **`coordinator.py`**: Core logic - `BermudaDataUpdateCoordinator` extends `DataUpdateCoordinator`. The `_async_update_data_internal()` method (runs every ~1s) processes BT data, calculates distances, assigns areas. This is the heart of the integration.
- **`bermuda_device.py`**: `BermudaDevice` class - internal device representation with distance calculation, area tracking, beacon/IRK handling
- **`bermuda_advert.py`**: Advertisement data from a specific scanner to a specific device
- **`bermuda_irk.py`**: Handles IRK resolution for iOS/Android rotating MAC addresses via `private_ble_device` integration
- **`const.py`**: All constants - configuration keys, timeouts, metadevice types, address types
- **`entity.py`**: `BermudaEntity` base class with rate-limiting (`_cached_ratelimit()`) to reduce database churn
- **Platform files** (`sensor.py`, `device_tracker.py`, `number.py`, `switch.py`): Create HA entities for tracked devices

## Development Workflow

### Setup & Running
```bash
./scripts/setup          # Initial setup, creates venv
./scripts/develop        # Run HA dev environment on port 9123
./scripts/test           # Run pytest with coverage (minimum 43%)
./scripts/lint           # Run pre-commit checks (ruff, black, etc)
```

**VS Code Debugger**: F5 launches debugpy - use this instead of `./scripts/develop`

**Devcontainer**: Full HA instance at `.devcontainer.json` - ports 8123 (HA), 5678 (debugger)

### Testing
- Test files in `tests/` mirror `custom_components/bermuda/` structure
- `conftest.py` provides fixtures: `mock_bermuda_entry`, `mock_bluetooth`, `skip_yaml_data_load`
- Coverage requirement: 43% minimum (see `scripts/test`)
- Use `pytest --durations=10` to identify slow tests

## Critical Patterns & Conventions

### MAC Address Handling
Always use `mac_norm()` from `util.py` for consistency:
```python
from .util import mac_norm
address = mac_norm(raw_mac)  # Returns lowercase xx:xx:xx:xx:xx:xx
```
Use `mac_explode_formats()` to check alternate formats (-, _, no separator)

### Address Types (`address_type` in BermudaDevice)
Check `BDADDR_TYPE_*` constants - devices are classified as:
- `BDADDR_TYPE_RANDOM_RESOLVABLE`: IRK-based rotating addresses
- `ADDR_TYPE_IBEACON`: iBeacon UUID (not a MAC48)
- `ADDR_TYPE_PRIVATE_BLE_DEVICE`: Meta-device for private BLE
- Determines how device is tracked and aggregated

### Rate-Limited Logging
Use `BermudaLogSpamLess` to avoid log spam:
```python
from .const import _LOGGER_SPAM_LESS
_LOGGER_SPAM_LESS.warning("unique_key", "This won't spam every cycle")
```
Default interval: `LOGSPAM_INTERVAL` (22 seconds)

### Sensor Updates & Caching
Entities use `_cached_ratelimit()` (in `entity.py`) to throttle state updates:
- Respects `CONF_UPDATE_INTERVAL` (default 10s)
- Immediately publishes decreasing distances (`fast_falling=True`)
- Prevents database bloat while maintaining responsiveness
- Force cache bust when `ref_power` changes

### Config Entry & Options
- Configuration stored in `config_entry.options` (mutable)
- `CONF_DEVICES`: List of addresses to track
- `CONFDATA_SCANNERS`: Persisted scanner data
- Use `async_reload_entry()` for option changes

### Services
`dump_devices` service (`service_dump_devices()` in coordinator):
- Returns full internal state for debugging/templating
- Parameters: `addresses`, `configured_devices`, `redact`
- Output format may change between versions (not a stable API)

## Common Gotchas

1. **Don't confuse BermudaDevice with HA Device Registry entries** - BermudaDevice is internal state
2. **Pruning**: Old device entries are auto-pruned (see `PRUNE_*` constants) - don't expect all discovered devices to persist
3. **Unique IDs**: Format is `{address}_{sensor_type}` for sensors, but meta-devices use beacon UUID or IRK
4. **Bluetooth backend changes**: HA's bluetooth integration timing (195s advert expiry) affects our logic
5. **Line length**: Project uses 120 chars, not HA's 88 (see `pyproject.toml`)

## Code Style
- **Formatter**: Black (120 char lines)
- **Linter**: Ruff with custom ignore list (see `pyproject.toml` - many ANN, D rules disabled)
- **Pre-commit**: Run `./scripts/lint` before committing
- **Imports**: Use `from __future__ import annotations` and TYPE_CHECKING blocks
- **Type hints**: Partial coverage (ANN rules mostly ignored for now)

## Integration Dependencies
- Depends on: `bluetooth_adapters`, `device_tracker`, `private_ble_device`
- Integrates with: Area Registry, Floor Registry, Device Registry
- Uses `bluetooth_data_tools` for monotonic timing

## Quick Reference
- Update loop: `UPDATE_INTERVAL` (1.05s) - fast internal processing
- Sensor updates: `CONF_UPDATE_INTERVAL` (10s default) - user-configurable throttle
- Distance timeout: `DISTANCE_TIMEOUT` (30s) - mark stale
- Device tracker timeout: `CONF_DEVTRACK_TIMEOUT` (30s default)
- Max area radius: `CONF_MAX_RADIUS` (20m default)

## When Modifying
- **Distance calculations**: See `rssi_to_metres()` in `util.py` and distance smoothing in `BermudaDevice`
- **Area assignment**: Logic in `_async_update_data_internal()` - uses closest scanner within `max_area_radius`
- **Adding sensors**: Extend platform files, add to `PLATFORMS` in `const.py`
- **Config options**: Add to `const.py` with `CONF_*` and `DEFAULT_*`, update config_flow forms
- **Bluetooth events**: Subscribe in coordinator's `__init__` via `bluetooth.async_register_callback()`
