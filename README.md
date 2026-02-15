![Bermuda Logo](img/logo@2x.png)

[![Open your Home Assistant instance and open a repository inside the Home Assistant Community Store.](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=agittins&repository=bermuda&category=Integration)

# Bermuda BLE Trilateration

- Track bluetooth devices by Area (Room) in [Home Assistant](https://home-assistant.io/), using [ESPHome](https://esphome.io/) [Bluetooth Proxies](https://esphome.io/components/bluetooth_proxy.html) and Shelly Gen2 or later devices.

- **NEW**: Trilateration support for precise (x, y, z) position tracking using multiple bluetooth proxies with configured coordinates!


[![GitHub Release][releases-shield]][releases]
[![GitHub Activity][commits-shield]][commits]
[![License][license-shield]](LICENSE)
[![HomeAssistant Minimum Version][haminverbadge]][haminver]
[![pre-commit][pre-commit-shield]][pre-commit]
[![Black][black-shield]][black]
[![hacs][hacsbadge]][hacs]
[![Project Maintenance][maintenance-shield]][user_profile]
[![Discord][discord-shield]][discord]
[![Community Forum][forum-shield]][forum]

[![GitHub Sponsors][sponsorsbadge]][sponsors]
[![BuyMeCoffee][buymecoffeebadge]][buymecoffee]
[![Patreon Sponsorship][patreonbadge]][patreon]


## What it does:

Bermuda aims to let you track any bluetooth device, and have Home Assistant tell you where in your house that device is. The only extra hardware you need are esp32 devices running esphome that act as bluetooth proxies. Alternatively, Shelly Plus devices can also perform this function.

- Area-based device location (ie, device-level room prescence) is working reasonably well.
- Creates sensors for Area and Distance for devices you choose
- Supports iBeacon devices, including those with randomised MAC addresses (like Android phones running HA Companion App)
- Supports IRK (resolvable keys) via the [Private BLE Device](https://www.home-assistant.io/integrations/private_ble_device/) core component. Once your iOS device (or Android!) is set up in Private BLE Device, it will automatically receive Bermuda sensors as well!
- Creates `device_tracker` entities for chosen devices, which can be linked to "Person"s for Home/Not Home tracking
- Configurable settings for rssi reference level, environmental attenuation, max tracking radius
- Provides a comprehensive json/yaml dump of devices and their distances from each bluetooth
  receiver, via the `bermuda.dump_devices` service.

## Trilateration (3D Position Tracking)

Bermuda now supports **trilateration** - calculating the precise (x, y, z) position of tracked devices using distance measurements from multiple bluetooth proxies with known coordinates.

### How it works:

- **4+ scanners**: Precise 3D positioning (x, y, z) with high confidence
- **3 scanners**: Reliable 2D positioning (x, y), potentially 3D if scanners are at different heights
- **2 scanners**: Ambiguous positioning with disambiguation using movement history and velocity limits
- **1 scanner**: Distance-only tracking with low confidence

### Setup:

1. **Configure scanner positions and room boundaries**: Use the Bermuda integration's "Bulk Import Map & Scanners" feature to import scanner positions and room polygons:

   a. Go to **Settings** → **Devices & Services** → **Bermuda BLE Trilateration**
   
   b. Click the **⋮ menu** (three dots) → **Bulk Import Map & Scanners**
   
   c. Paste JSON configuration with the physical locations of your bluetooth proxies and room polygons:

```json
{
  "floors": [
    {
      "id": "ground",
      "name": "Ground Floor",
      "bounds": [[0, 0, 0], [10, 8, 3]]
    }
  ],
  "rooms": [
    {
      "id": "living_room",
      "name": "Living Room",
      "floor": "ground",
      "area_id": "living_room",
      "points": [
        [0.0, 0.0],
        [4.0, 0.0],
        [4.0, 3.0],
        [0.0, 3.0]
      ]
    },
    {
      "id": "kitchen",
      "name": "Kitchen",
      "floor": "ground",
      "area_id": "kitchen",
      "points": [
        [0.0, 3.0],
        [4.0, 3.0],
        [4.0, 8.0],
        [0.0, 8.0]
      ]
    }
  ],
  "nodes": [
    {
      "id": "AA:BB:CC:DD:EE:FF",
      "name": "Living Room Proxy",
      "point": [2.0, 1.5, 1.0]
    },
    {
      "id": "11:22:33:44:55:66",
      "name": "Bedroom Proxy",
      "point": [7.0, 2.5, 1.0]
    }
  ]
}
```

**Key fields:**
- `floors`: Define floor boundaries for multi-floor homes
  - `bounds`: [[min_x, min_y, min_z], [max_x, max_y, max_z]]
- `rooms`: Define room polygons for automatic area detection
  - `area_id`: Must match your Home Assistant Area ID
  - `points`: Polygon vertices in [x, y] coordinates (clockwise or counter-clockwise)
- `nodes`: Scanner positions
  - `id`: MAC address of your bluetooth proxy (must match exactly)
  - `point`: [x, y, z] coordinates in meters

**Coordinate system:**
- Choose any origin point (e.g., bottom-left corner of your floorplan)
- x, y: horizontal position in meters
- z: height above floor (typically 0.5-2.0 meters for wall-mounted proxies)

   d. Click **"Replace All"** (clears existing config) or **"Merge"** (adds to existing)
   
   e. Submit and reload the integration for changes to take effect

2. **Enable the Position sensor** for your tracked devices in the entity settings (disabled by default until scanner positions are configured)

### Position Sensor:

The Position sensor provides:
- **State**: Formatted coordinates like `(2.5, 3.1, 1.0)`
- **Attributes**:
  - `x`, `y`, `z`: Individual coordinate values
  - `confidence`: Percentage (0-100) indicating position accuracy
  - `method`: Algorithm used (`1-scanner`, `2-scanner`, `3-scanner`, `4+scanner`)
  - `scanner_count`: Number of scanners used in calculation
  - `room_id`, `room_name`, `floor_id`: Detected room/floor (when rooms are configured)

### Automatic Area Detection:

When rooms are defined in `scanner_positions.json`, Bermuda automatically:
- **Calculates device position** from scanner distances
- **Determines which room** the device is in using point-in-polygon detection
- **Updates the Area sensor** to match the detected room
- **Overrides distance-based area assignment** for more accurate room presence

This means your device's Area sensor will update based on precise position rather than just "closest scanner", providing much more accurate room-level presence detection!

**Configuration:**
- `trilateration_override_area` (default: true) - Use position to set device area
- `trilateration_area_min_confidence` (default: 30%) - Minimum confidence to override area

### Advanced RSSI Filtering (NEW) 🎯

Bermuda now includes adaptive RSSI filtering inspired by ESPresence, providing robust outlier rejection for more accurate distance estimates in noisy BLE environments:

- **Adaptive Percentile IQR Filtering**: Uses Tukey's outlier detection method to statistically reject RSSI spikes
- **Variance-Based Trilateration Weighting**: Prioritizes scanners with stable signal quality  
- **Automatic Buffer Sizing**: Adapts to different device advertisement rates

**See [RSSI_FILTERING.md](RSSI_FILTERING.md) for complete documentation.**

**Quick configuration (optional - defaults work well):**
```yaml
rssi_filter_mode: "median_iqr"  # or "legacy" for old behavior
iqr_coefficient: 1.5  # 1.0-2.0, controls outlier strictness
trilateration_use_variance_weighting: true  # weight scanners by signal quality
```

### Tips:

- Place proxies at room corners for best coverage
- Height variation (z-coordinate) improves 3D accuracy
- Use at least 3 proxies for reliable positioning
- Define room polygons matching your Home Assistant Areas for automatic area detection
- The `confidence` attribute helps you filter unreliable positions in automations
- **Export your config**: Use **⋮ menu** → **Export Map & Scanners (JSON)** to save a backup
- **Use scanner_positions.json.example** as a template when creating your configuration

See [the Wiki](https://github.com/agittins/bermuda/wiki/) for detailed setup guides and troubleshooting.

## Recommended ESPHome Scanner Configuration

For reliable BLE proxy operation, your ESPHome configs should include three critical stability features: **API-controlled scan lifecycle**, **ESP-IDF version pinning**, and **safe mode**. Without these, scanners can freeze or become unresponsive after network interruptions.

### Minimal recommended config:

```yaml
esp32:
  board: esp32dev
  framework:
    type: esp-idf
    version: 5.3.2  # pin to avoid known BLE issues in 5.5.x
    sdkconfig_options:
      CONFIG_BT_BLE_42_FEATURES_SUPPORTED: y
      CONFIG_ESP_TASK_WDT_TIMEOUT_S: "10"

logger:
  baud_rate: 0  # free CPU/memory by disabling serial logging
  level: WARN

api:
  encryption:
    key: ${api_key}
  on_client_connected:
    - esp32_ble_tracker.start_scan:
        continuous: true
  on_client_disconnected:
    if:
      condition:
        not:
          api.connected:
      then:
        - esp32_ble_tracker.stop_scan:

wifi:
  ssid: !secret wifi_ssid
  password: !secret wifi_password
  power_save_mode: NONE

esp32_ble_tracker:
  scan_parameters:
    continuous: false  # only scan when HA API is connected
    active: true
    interval: 320ms
    window: 300ms   # ~94% duty cycle for maximum BLE coverage

bluetooth_proxy:
  active: true

safe_mode:  # allows OTA recovery if firmware crashes repeatedly
```

### Why these settings matter:

- **`continuous: false` + API lifecycle events**: Prevents the ESP32 from running BLE scans before WiFi is connected. Without this, a network disruption (WiFi AP restart, HA restart) can leave the scanner running BLE with no WiFi, exhausting resources until it becomes unresponsive.
- **`version: 5.3.2`** (ESP-IDF pin): ESP-IDF 5.5.x has known issues causing BLE proxy freezes on classic ESP32 boards. Version 5.3.2 was the stable default through ESPHome 2025.6–2025.10.
- **`safe_mode:`**: If the device crashes repeatedly, it boots into a minimal safe mode where OTA updates still work — no need to physically access the device with a USB cable.
- **`power_save_mode: NONE`**: Disables WiFi power saving for consistent connectivity.
- **`interval: 320ms` / `window: 300ms`**: 94% BLE listen duty cycle (vs the default ~9%), dramatically improving device detection.
- **`baud_rate: 0`**: Disabling serial UART logging frees CPU cycles and memory for BLE operations.
- **`CONFIG_BT_BLE_42_FEATURES_SUPPORTED: y`**: Enables BLE 4.2 features for better bluetooth performance on original ESP32 boards.

### Board recommendations:

- **ESP32-S3**: Best choice — dual-core, BLE 5.0, good memory. Boards like M5Stack Atom S3 work well.
- **ESP32 (original)**: Dual-core, reliable with the config above. Most common and cheapest option (D1-Mini32, etc).
- **ESP32-C3**: Single-core, can be problematic under load. If using C3, the API lifecycle control above is especially important.
- **ESP32-C5/C6**: Newer chips with BLE 5.0 and WiFi 6. Good performance but less community testing.

### Also see:

- [agittins/bermuda-proxies](https://github.com/agittins/bermuda-proxies) — Official ESPHome packages for Bermuda with ready-to-use configs
- [ESPHome Bluetooth Proxy docs](https://esphome.io/components/bluetooth_proxy.html)

## What you need:

- Home Assistant. The current release of Bermuda requires at least ![haminverbadge]
- One or more devices providing bluetooth proxy information to HA using HA's bluetooth backend. These can be:
  - ESPHome devices with the `bluetooth_proxy` component enabled. I like the D1-Mini32 boards because they're cheap and easy to deploy.
  - Shelly Plus or later devices with Bluetooth proxying enabled in the Shelly integration.
  - USB Bluetooth on your HA host. This is not ideal, since they do not timestamp the advertisement packets and finding a well-supported usb bluetooth adaptor is non-trivial. However they can be used for simple "Home/Not Home" tracking, and basic Area distance support is enabled currently.

- Some bluetooth BLE devices you want to track. Phones, smart watches, beacon tiles, thermometers etc.

- Bermuda! I strongly recommend installing Bermuda via HACS:
  [![Open your Home Assistant instance and open a repository inside the Home Assistant Community Store.](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=agittins&repository=bermuda&category=Integration)

## Documentation and help

[The Wiki](https://github.com/agittins/bermuda/wiki/) is the primary and official source of information for setting up Bermuda.

[Discussions](https://github.com/agittins/bermuda/discussions/) contain both official and user-contributed guides, how-tos and general Q&A.

[HA Community Thread for Bermuda](https://community.home-assistant.io/t/bermuda-bluetooth-ble-room-presence-and-tracking-custom-integration/625780/1) contains a *wealth* of information from and for users of Bermuda, and is where many folk first ask for assistance in setting up.

## Screenshots

After installing, the integration should be visible in Settings, Devices & Services

![The integration, in Settings, Devices & Services](img/screenshots/integration.png)

Press the `CONFIGURE` button to see the configuration dialog. At the bottom is a field
where you can enter/list any bluetooth devices the system can see. Choosing devices
will add them to the configured devices list and creating sensor entities for them. See [How Do The Settings Work?](#how-do-the-settings-work) for more info.

![Bermuda integration configuration option flow](img/screenshots/configuration.png)

Choosing the device screen shows the current sensors and other info. Note that there are extra sensors in the "not shown" section that are disabled by default (the screenshot shows several of these enabled already). You can edit the properties of these to enable them for more detailed data on your device locations. This is primarily intended for troubleshooting or development, though.

![Screenshot of device information view](img/screenshots/deviceinfo.png)

The sensor information also includes attributes area name and id, relevant MAC addresses
etc.

![Bermuda sensor information](img/screenshots/sensor-info.png)

In Settings, People, you can define any Bermuda device to track home/away status
for any person/user.

![Assign a Bermuda sensor for Person tracking](img/screenshots/person-tracker.png)

## FAQ

See [The FAQ](https://github.com/agittins/bermuda/wiki/FAQ) in the Wiki!

## Hacking tips

Wanna improve this? Awesome! Bear in mind this is my first ever HA
integration, and I'm much more greybeard sysadmin than programmer, so ~~if~~where
I'm doing stupid things I really would welcome some improvements!

You can start by using the service `bermuda.dump_devices` to examine the
internal state.

### Using `bermuda.dump_devices` service

Just calling the service `bermuda.dump_devices` will give you a full dump of the internal
data structures that bermuda uses to track and calculate its state. This can be helpful
for working out what's going on and troubleshooting, or to use if you have a very custom
need that you can solve with template sensors etc.

If called with no parameters, the service will return all data. parameters are available
which let you limit or reformat the resulting data to make it easier to work with. In particular
the `addresses` parameter is helpful to only return data relevant for one or more MAC addresses
(or iBeacon UUIDs).
See the information on parameters in the `Services` page in Home Assistant, under `Developer Tools`.

Important: If you decide to use the results of this call for your own templates etc, bear in mind that
the format might change in any release, and won't necessarily be considered a "breaking change".
This is beacuse the structure is used internally, rather than being a published API. That said, efforts will be made
to indicate in the release notes if fields in the structure are renamed or moved, but not for adding new
items.

## Prior Art

The `bluetooth_tracker` and `ble_tracker` integrations are only built to give a "home/not home"
determination, and don't do "Area" based location. (nb: "Zones" are places outside the
home, while "Areas" are rooms/areas inside the home). I wanted to be free to experiment with
this in ways that might not suit core, but hopefully at least some of this could find
a home in the core codebase one day.

The "monitor" script uses standalone Pi's to gather bluetooth data and then pumps it into
MQTT. It doesn't use the `bluetooth_proxy` capabilities which I feel are the future of
home bluetooth networking (well, it is for my home, anyway!).

ESPresense looks cool, but I don't want to dedicate my nodes to non-esphome use, and again
it doesn't leverage the bluetooth proxy features now in HA. I am probably reinventing
a fair amount of ESPresense's wheel.

## Installation

You can install Bermuda by opening HACS on your Home Assistant instance and searching for "Bermuda".
Alternatively you can click the button below to be automatically redirected.

[![Open your Home Assistant instance and open a repository inside the Home Assistant Community Store.](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=agittins&repository=bermuda&category=Integration)

You should now be able to add the `Bermuda BLE Trilateration` integration. Once you have done that,
you need to restart Home Assistant, then in `Settings`, `Devices & Services` choose `Add Integration`
and search for `Bermuda BLE Trilateration`. It's possible that it will autodetect for you just by
noticing nearby bluetooth devices.

Once the integration is added, you need to set up your devices by clicking `Configure` in `Devices and Services`,
`Bermuda BLE Trilateration`.

In the `Configuration` dialog, you can choose which bluetooth devices you would like the integration to track.

You can manually install Bermuda by doing the following:

1. Using the tool of choice open the directory (folder) for your HA configuration (where you find `configuration.yaml`).
2. If you do not have a `custom_components` directory (folder) there, you need to create it.
3. In the `custom_components` directory (folder) create a new folder called `bermuda`.
4. Download _all_ the files from the `custom_components/bermuda/` directory (folder) in this repository.
5. Place the files you downloaded in the new directory (folder) you created.
6. Restart Home Assistant
7. In the HA UI go to "Configuration" -> "Integrations" click "+" and search for "Bermuda BLE Trilateration"

<!---->

## Contributions are welcome!

If you want to contribute to this please read the [Contribution guidelines](CONTRIBUTING.md)

## Credits

This project was generated from [@oncleben31](https://github.com/oncleben31)'s [Home Assistant Custom Component Cookiecutter](https://github.com/oncleben31/cookiecutter-homeassistant-custom-component) template.

Code template was mainly taken from [@Ludeeus](https://github.com/ludeeus)'s [integration_blueprint][integration_blueprint] template
[Cookiecutter User Guide](https://cookiecutter-homeassistant-custom-component.readthedocs.io/en/stable/quickstart.html)\*\*

---

[integration_blueprint]: https://github.com/custom-components/integration_blueprint

[black]: https://github.com/psf/black
[black-shield]: https://img.shields.io/badge/code%20style-black-000000.svg?style=for-the-badge

[buymecoffee]: https://www.buymeacoffee.com/AshleyGittins
[buymecoffeebadge]: https://img.shields.io/badge/buy%20me%20a%20coffee-Caffeinate-green.svg?style=for-the-badge

[commits-shield]: https://img.shields.io/github/commit-activity/y/agittins/bermuda.svg?style=for-the-badge
[commits]: https://github.com/agittins/bermuda/commits/main

[hacs]: https://hacs.xyz
[hacsbadge]: https://img.shields.io/badge/HACS-Default-green.svg?style=for-the-badge

[haminver]: https://github.com/agittins/bermuda/commits/main/hacs.json
[haminverbadge]: https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fgithub.com%2Fagittins%2Fbermuda%2Fraw%2Fmain%2Fhacs.json&query=%24.homeassistant&style=for-the-badge&logo=homeassistant&logoColor=%2311BDF2&label=Minimum%20HA%20Version

[discord]: https://discord.gg/Qa5fW2R
[discord-shield]: https://img.shields.io/discord/330944238910963714.svg?style=for-the-badge

[exampleimg]: example.png
[forum-shield]: https://img.shields.io/badge/community-forum-brightgreen.svg?style=for-the-badge
[forum]: https://community.home-assistant.io/

[license-shield]: https://img.shields.io/github/license/agittins/bermuda.svg?style=for-the-badge
[maintenance-shield]: https://img.shields.io/badge/maintainer-%40agittins-blue.svg?style=for-the-badge

[patreon]: https://patreon.com/AshGittins
[patreonbadge]: https://img.shields.io/badge/Patreon-Sponsor-green?style=for-the-badge

[pre-commit]: https://github.com/pre-commit/pre-commit
[pre-commit-shield]: https://img.shields.io/badge/pre--commit-enabled-brightgreen?style=for-the-badge

[sponsorsbadge]: https://img.shields.io/github/sponsors/agittins?style=for-the-badge&label=GitHub%20Sponsors&color=green
[sponsors]: https://github.com/sponsors/agittins

[releases-shield]: https://img.shields.io/github/release/agittins/bermuda.svg?style=for-the-badge
[releases]: https://github.com/agittins/bermuda/releases
[user_profile]: https://github.com/agittins
