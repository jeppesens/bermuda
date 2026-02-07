# CLAUDE.md - Bermuda BLE Trilateration

## Project Overview

Bermuda is a **Home Assistant custom integration** for Bluetooth device tracking and room-level localization. It uses BLE advertisements from ESPHome Bluetooth Proxies, Shelly devices, and other scanners to determine which area/room a tracked device is in. Supports iBeacon devices, Private BLE Devices (via IRK resolution), and standard BLE transmitters.

- **Domain**: `bermuda`
- **License**: MIT
- **Min HA version**: 2025.3
- **Python target**: 3.12+
- **HACS**: Yes (custom component distributed via HACS)

## Repository Structure

```
custom_components/bermuda/    # Main integration source
  coordinator.py              # Core: DataUpdateCoordinator - fetches/processes BLE data
  bermuda_device.py           # BermudaDevice: internal device representation
  bermuda_advert.py           # BermudaAdvert: device-scanner RSSI relationship
  bermuda_irk.py              # IRK Manager: Identity Resolving Key resolution
  config_flow.py              # Config and options flow UI
  sensor.py                   # Sensor platform (area, distance, RSSI, floor)
  device_tracker.py           # Device tracker platform (home/away)
  number.py                   # Number platform (per-device RSSI calibration)
  entity.py                   # Base entity classes
  const.py                    # All constants, config keys, defaults
  util.py                     # Utility functions (MAC normalization, RSSI math)
  __init__.py                 # Integration setup and migration
  services.yaml               # Service definitions
  manifest.json               # HA integration manifest
  log_spam_less.py            # Rate-limited logging helper
  diagnostics.py              # Diagnostic data export
  binary_sensor.py            # Stub - not yet implemented
  switch.py                   # Stub - not yet implemented
  translations/               # Multi-language UI strings
  manufacturer_identification/ # BLE manufacturer data (YAML)
tests/                        # Test suite
  conftest.py                 # Fixtures (mock bluetooth, config entries)
  test_init.py                # Integration setup/unload tests
  test_config_flow.py         # Config flow tests
  test_bermuda_device.py      # BermudaDevice tests
  test_bermuda_advert.py      # BermudaAdvert tests
  test_util.py                # Utility function tests
  test_switch.py              # Switch platform tests
scripts/                      # Dev helper scripts (setup, develop, lint, test)
```

## Build & Test Commands

```bash
# Run full test suite with coverage
pytest tests

# Run tests matching CI (with timeout + parallel)
pytest --timeout=9 -n auto -p no:sugar tests

# Run a specific test file
pytest tests/test_util.py

# Run pre-commit linting (ruff format + lint)
pre-commit run --all-files

# Dev helper scripts
./scripts/setup    # Install dev dependencies
./scripts/develop  # Development environment setup
./scripts/lint     # Run pre-commit hooks
./scripts/test     # Run pytest with coverage
```

## Code Style & Linting

- **Ruff** is the primary linter and formatter (configured in `pyproject.toml`)
  - Line length: **120 characters**
  - Target: Python 3.12
  - Rule selection: `ALL` with specific ignores (see `pyproject.toml [tool.ruff.lint]`)
  - Ruff rules are relaxed for test files (`tests/**/*.py` ignores `ALL`)
- **Pre-commit hooks** (`.pre-commit-config.yaml`): YAML check, large file check, end-of-file fixer, trailing whitespace, ruff lint with `--fix --unsafe-fixes`, ruff format
- **Coverage requirement**: 100% (`setup.cfg [coverage:report] fail_under = 100`)
- **Async**: All HA platform code uses `async`/`await`. Pytest uses `asyncio_mode = auto`

## Key Architecture Patterns

### DataUpdateCoordinator Pattern
`BermudaDataUpdateCoordinator` (in `coordinator.py`) is the core engine. It:
- Runs every 1.05 seconds (`UPDATE_INTERVAL`)
- Iterates all HA bluetooth advertisements
- Creates/updates `BermudaDevice` and `BermudaAdvert` objects
- Runs area determination algorithm (closest scanner by RSSI)
- Emits dispatcher signals (`SIGNAL_DEVICE_NEW`, `SIGNAL_SCANNERS_CHANGED`) for entity creation

### Entity Hierarchy
- `BermudaEntity` (extends `CoordinatorEntity`) - base for per-device entities
- `BermudaGlobalEntity` (extends `CoordinatorEntity`) - for integration-wide entities
- Entities use `_cached_ratelimit()` for update throttling
- Entity state updates come from the coordinator via the standard HA pattern

### Device Model
- `BermudaDevice` - represents any BLE device (transmitter or scanner/receiver)
- `BermudaAdvert` - represents the relationship between a device and a scanner (RSSI, distance, smoothing)
- Meta-devices exist for iBeacon and Private BLE devices (tracked by UUID/IRK instead of MAC)

### Distance Calculation Pipeline
```
Raw RSSI -> rssi_to_metres() (path loss formula) -> smoothing average -> rate-limited sensor update
```

### Config Flow
- Single-instance integration (`single_config_entry` in `config_flow.py`)
- Multi-step options flow: global options + per-device configuration
- Configuration stored in `config_entry.options` dict

## HA Import Conventions

Follow Home Assistant's standard import aliases (enforced by ruff):
```python
import homeassistant.helpers.area_registry as ar
import homeassistant.helpers.config_validation as cv
import homeassistant.helpers.device_registry as dr
import homeassistant.helpers.entity_registry as er
import homeassistant.helpers.floor_registry as fr
import voluptuous as vol
import homeassistant.util.dt as dt_util
```

## Testing Conventions

- Tests use `pytest-homeassistant-custom-component` for HA-specific fixtures
- `conftest.py` provides: `mock_setup_entry`, `mock_config_entry`, bluetooth mocks
- Async tests are automatic (no `@pytest.mark.asyncio` needed due to `asyncio_mode = auto`)
- Test timeout: 9 seconds per test in CI
- All new code must maintain 100% test coverage

## Versioning

- `const.py` `VERSION = "0.0.0"` in the repo (development sentinel)
- Actual version is injected by GitHub Actions release workflow from git tags
- `manifest.json` version is also updated during release
- Never manually change `VERSION` in `const.py` or `manifest.json`

## CI/CD

GitHub Actions workflows in `.github/workflows/`:
- **tests.yaml**: Linting (pre-commit/ruff), HACS validation, Hassfest validation, pytest (Python 3.13, uv package manager)
- **release.yaml**: Version injection, zip packaging, Sigstore signing
- Runs on push to main/master/dev, PRs, and daily schedule

## Key Constants (from `const.py`)

| Constant | Value | Purpose |
|----------|-------|---------|
| `UPDATE_INTERVAL` | 1.05s | BLE data processing cycle |
| `DISTANCE_TIMEOUT` | 30s | Mark distance as stale |
| `DISTANCE_INFINITE` | 999 | Unknown/infinite distance |
| `DEFAULT_MAX_RADIUS` | 20m | Max area detection radius |
| `DEFAULT_MAX_VELOCITY` | 3 m/s | Ignore unrealistic movement |
| `DEFAULT_REF_POWER` | -55 dBm | Signal at 1 metre reference |
| `DEFAULT_SMOOTHING_SAMPLES` | 20 | Distance averaging window |
| `DEFAULT_DEVTRACK_TIMEOUT` | 30s | Device tracker away timeout |
| `DEFAULT_UPDATE_INTERVAL` | 10s | Sensor update interval |

## Common Development Tasks

### Adding a new sensor type
1. Define sensor description in `sensor.py` (follow existing `BermudaSensor*` patterns)
2. Add entity class extending `BermudaEntity`
3. Register in `async_setup_entry()` via dispatcher signal callback
4. Add corresponding test in `tests/`

### Adding a new configuration option
1. Add `CONF_*` and `DEFAULT_*` constants in `const.py`
2. Add to config flow schema in `config_flow.py`
3. Add translation strings in `translations/en.json`
4. Read option in `coordinator.py` via `self.config_entry.options.get()`

### Working with BLE data
- All bluetooth data comes through HA's `bluetooth` component
- Scanner devices are identified by having entries in the HA device registry with bluetooth connections
- RSSI-to-distance conversion uses the log-distance path loss model in `util.py`
