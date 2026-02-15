"""DataUpdateCoordinator for Bermuda bluetooth data."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from typing import TYPE_CHECKING
from typing import cast

import aiofiles
import voluptuous as vol
import yaml
from bluetooth_data_tools import monotonic_time_coarse
from habluetooth import BaseHaScanner
from homeassistant.components import bluetooth
from homeassistant.components.bluetooth.api import _get_manager
from homeassistant.components.persistent_notification import async_create as async_create_notification
from homeassistant.components.persistent_notification import async_dismiss as async_dismiss_notification
from homeassistant.const import MAJOR_VERSION as HA_VERSION_MAJ
from homeassistant.const import MINOR_VERSION as HA_VERSION_MIN
from homeassistant.const import Platform
from homeassistant.core import Event
from homeassistant.core import HomeAssistant
from homeassistant.core import ServiceCall
from homeassistant.core import ServiceResponse
from homeassistant.core import SupportsResponse
from homeassistant.core import callback
from homeassistant.helpers import area_registry as ar
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers import entity_registry as er
from homeassistant.helpers import floor_registry as fr
from homeassistant.helpers import issue_registry as ir
from homeassistant.helpers.device_registry import EVENT_DEVICE_REGISTRY_UPDATED
from homeassistant.helpers.event import async_track_time_interval
from homeassistant.helpers.device_registry import EventDeviceRegistryUpdatedData
from homeassistant.helpers.dispatcher import async_dispatcher_send
from homeassistant.helpers.update_coordinator import DataUpdateCoordinator
from homeassistant.util.dt import get_age
from homeassistant.util.dt import now

from .bermuda_device import BermudaDevice
from .bermuda_irk import BermudaIrkManager
from .const import _LOGGER
from .const import _LOGGER_SPAM_LESS
from .const import ADDR_TYPE_PRIVATE_BLE_DEVICE
from .const import AREA_MAX_AD_AGE
from .const import AUTO_CALIBRATION_MIN_SCANNER_PAIRS
from .const import BDADDR_TYPE_NOT_MAC48
from .const import BDADDR_TYPE_RANDOM_RESOLVABLE
from .const import CALIBRATION_NOTIFICATION_COOLDOWN
from .const import CONF_ATTENUATION
from .const import CONF_AUTO_CALIBRATION_ENABLED
from .const import CONF_AUTO_CALIBRATION_INTERVAL_HOURS
from .const import CONF_AUTO_CALIBRATION_METHOD
from .const import CONF_AUTO_CALIBRATION_SAMPLES
from .const import CONF_DEVICES
from .const import CONF_DEVTRACK_TIMEOUT
from .const import CONF_ENABLE_TRILATERATION
from .const import CONF_MAX_RADIUS
from .const import CONF_MAX_TRILATERATION_SCANNERS
from .const import CONF_MAX_VELOCITY
from .const import CONF_MIN_TRILATERATION_SCANNERS
from .const import CONF_REF_POWER
from .const import CONF_RSSI_OFFSETS
from .const import CONF_SMOOTHING_SAMPLES
from .const import CONF_TRILATERATION_AREA_MIN_CONFIDENCE
from .const import CONF_TRILATERATION_DEBUG
from .const import CONF_TRILATERATION_OVERRIDE_AREA
from .const import CONF_UPDATE_INTERVAL
from .const import CONFDATA_CALIBRATION_QUALITY
from .const import CONFDATA_CALIBRATION_STATUS
from .const import CONFDATA_FLOORS
from .const import CONFDATA_LAST_CALIBRATION_TIME
from .const import CONFDATA_ROOMS
from .const import CONFDATA_SCANNER_POSITIONS
from .const import DEFAULT_ATTENUATION
from .const import DEFAULT_AUTO_CALIBRATION_ENABLED
from .const import DEFAULT_AUTO_CALIBRATION_INTERVAL_HOURS
from .const import DEFAULT_AUTO_CALIBRATION_METHOD
from .const import DEFAULT_DEVTRACK_TIMEOUT
from .const import DEFAULT_MAX_RADIUS
from .const import DEFAULT_MAX_TRILATERATION_SCANNERS
from .const import DEFAULT_MAX_VELOCITY
from .const import DEFAULT_REF_POWER
from .const import DEFAULT_SMOOTHING_SAMPLES
from .const import DEFAULT_TRILATERATION_DEBUG
from .const import DEFAULT_UPDATE_INTERVAL
from .const import DOMAIN
from .const import DOMAIN_PRIVATE_BLE_DEVICE
from .const import METADEVICE_IBEACON_DEVICE
from .const import METADEVICE_TYPE_IBEACON_SOURCE
from .const import METADEVICE_TYPE_PRIVATE_BLE_SOURCE
from .const import NOTIFICATION_ID_CALIBRATION_INSUFFICIENT
from .const import NOTIFICATION_ID_CALIBRATION_NO_BROADCASTING
from .const import CONF_KALMAN_MAX_VELOCITY
from .const import CONF_KALMAN_MEASUREMENT_NOISE
from .const import CONF_KALMAN_PROCESS_NOISE
from .const import CONF_YAML_CONFIG_FILE
from .const import CONFDATA_NODE_FLOORS
from .const import DEFAULT_KALMAN_MAX_VELOCITY
from .const import DEFAULT_KALMAN_MEASUREMENT_NOISE
from .const import DEFAULT_KALMAN_PROCESS_NOISE
from .const import NOTIFICATION_ID_CALIBRATION_SUCCESS
from .const import PRUNE_MAX_COUNT
from .const import PRUNE_TIME_DEFAULT
from .const import PRUNE_TIME_INTERVAL
from .const import PRUNE_TIME_KNOWN_IRK
from .const import PRUNE_TIME_REDACTIONS
from .const import PRUNE_TIME_UNKNOWN_IRK
from .const import REPAIR_SCANNER_WITHOUT_AREA
from .const import SAVEOUT_COOLDOWN
from .const import SIGNAL_DEVICE_NEW
from .const import SIGNAL_SCANNERS_CHANGED
from .const import TRILATERATION_POSITION_TIMEOUT
from .const import UPDATE_INTERVAL
from .kalman import KalmanFilterSettings
from .kalman import KalmanLocation
from .trilateration import calculate_position
from .trilateration import find_room_for_position
from .util import mac_explode_formats
from .util import mac_norm
from .util import validate_scanners_for_trilateration

if TYPE_CHECKING:
    from habluetooth import BluetoothServiceInfoBleak
    from homeassistant.components.bluetooth import BluetoothChange
    from homeassistant.components.bluetooth.manager import HomeAssistantBluetoothManager

    from . import BermudaConfigEntry
    from .bermuda_advert import BermudaAdvert

Cancellable = Callable[[], None]

# Using "if" instead of "min/max" triggers PLR1730, but when
# split over two lines, ruff removes it, then complains again.
# so we're just disabling it for the whole file.
# https://github.com/astral-sh/ruff/issues/4244
# ruff: noqa: PLR1730


def _slugify_name(name: str) -> str:
    """Convert a name to a slug ID (lowercase, spaces to hyphens)."""
    import re as _re
    return _re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


class BermudaDataUpdateCoordinator(DataUpdateCoordinator):
    """
    Class to manage fetching data from the Bluetooth component.

    Since we are not actually using an external API and only computing local
    data already gathered by the bluetooth integration, the update process is
    very cheap, and the processing process (currently) rather cheap.

    TODO / IDEAS:
    - when we get to establishing a fix, we can apply a path-loss factor to
      a calculated vector based on previously measured losses on that path.
      We could perhaps also fine-tune that with real-time measurements from
      fixed beacons to compensate for environmental factors.
    - An "obstruction map" or "radio map" could provide field strength estimates
      at given locations, and/or hint at attenuation by counting "wall crossings"
      for a given vector/path.

    """

    def __init__(
        self,
        hass: HomeAssistant,
        entry: BermudaConfigEntry,
    ) -> None:
        """Initialize."""
        self.platforms = []
        self.config_entry = entry

        self.sensor_interval = entry.options.get(CONF_UPDATE_INTERVAL, DEFAULT_UPDATE_INTERVAL)

        # set some version flags
        self.hass_version_min_2025_2 = HA_VERSION_MAJ > 2025 or (HA_VERSION_MAJ == 2025 and HA_VERSION_MIN >= 2)
        # when habasescanner.discovered_device_timestamps became a public method.
        self.hass_version_min_2025_4 = HA_VERSION_MAJ > 2025 or (HA_VERSION_MAJ == 2025 and HA_VERSION_MIN >= 4)

        # ##### Redaction Data ###
        #
        # match/replacement pairs for redacting addresses
        self.redactions: dict[str, str] = {}
        # Any remaining MAC addresses will be replaced with this. We define it here
        # so we can compile it once. MAC addresses may have [:_-] separators.
        self._redact_generic_re = re.compile(
            r"(?P<start>[0-9A-Fa-f]{2})[:_-]([0-9A-Fa-f]{2}[:_-]){4}(?P<end>[0-9A-Fa-f]{2})"
        )
        self._redact_generic_sub = r"\g<start>:xx:xx:xx:xx:\g<end>"

        self.stamp_redactions_expiry: float | None = None

        self.update_in_progress: bool = False  # A lock to guard against huge backlogs / slow processing
        self.stamp_last_update: float = 0  # Last time we ran an update, from monotonic_time_coarse()
        self.stamp_last_update_started: float = 0
        self.stamp_last_prune: float = 0  # When we last pruned device list
        self.stamp_last_position_load: float = 0  # Track when scanner positions were last loaded

        self.member_uuids = {}
        self.company_uuids = {}

        super().__init__(
            hass,
            _LOGGER,
            name=DOMAIN,
            update_interval=timedelta(seconds=UPDATE_INTERVAL),
        )

        self._waitingfor_load_manufacturer_ids = True
        entry.async_create_background_task(
            hass, self.async_load_manufacturer_ids(), "Load Bluetooth IDs", eager_start=True
        )

        self._manager: HomeAssistantBluetoothManager = _get_manager(hass)  # instance of the bluetooth manager
        self._ha_scanners: set[BaseHaScanner] = set()  # Links to the backend scanners
        self._ha_scanner_timestamps: dict[str, dict[str, float]] = {}  # scanner_address, device_address, stamp
        self._scanner_list: set[str] = set()
        self._scanners: set[BermudaDevice] = set()  # Set of all in self.devices that is_scanner=True
        self.irk_manager = BermudaIrkManager()

        # Map/floor plan data for trilateration
        self.map_floors: dict[str, dict] = {}
        self.map_rooms: dict[str, dict] = {}

        self.ar = ar.async_get(self.hass)
        self.er = er.async_get(self.hass)
        self.dr = dr.async_get(self.hass)
        self.fr = fr.async_get(self.hass)
        self.have_floors: bool = self.init_floors()

        self._scanners_without_areas: list[str] | None = None  # Tracks any proxies that don't have an area assigned.

        # Track the list of Private BLE devices, noting their entity id
        # and current "last address".
        self.pb_state_sources: dict[str, str | None] = {}

        self.metadevices: dict[str, BermudaDevice] = {}

        self._ad_listener_cancel: Cancellable | None = None

        # Tracks the last stamp that we *actually* saved our config entry. Mostly for debugging,
        # we use a request stamp for tracking our add_job request.
        self.last_config_entry_update: float = 0  # Stamp of last *save-out* of config.data

        # We want to delay the first save-out, since it takes a few seconds for things
        # to stabilise. So set the stamp into the future.
        self.last_config_entry_update_request = (
            monotonic_time_coarse() + SAVEOUT_COOLDOWN
        )  # Stamp for save-out requests

        # AJG 2025-04-23 Disabling, see the commented method below for notes.
        # self.config_entry.async_on_unload(self.hass.bus.async_listen(EVENT_STATE_CHANGED, self.handle_state_changes))

        # First time around we freshen the restored scanner info by
        # forcing a scan of the captured info.
        self._scanner_init_pending = True
        self._scanner_positions_loaded = False  # Track if we've loaded scanner positions

        self._seed_configured_devices_done = False

        # First time go through the private ble devices to see if there's
        # any there for us to track.
        self._do_private_device_init = True

        # Auto-calibration tracking
        self._last_calibration_time: float = 0  # Monotonic timestamp of last calibration run
        self._last_calibration_notification: float = 0  # Cooldown for insufficient data notifications
        self._calibration_in_progress: bool = False  # Prevent concurrent calibration runs

        # Listen for changes to the device registry and handle them.
        # Primarily for changes to scanners and Private BLE Devices.
        self.config_entry.async_on_unload(
            self.hass.bus.async_listen(EVENT_DEVICE_REGISTRY_UPDATED, self.handle_devreg_changes)
        )

        # Register periodic auto-calibration task (checks every hour)
        self.config_entry.async_on_unload(
            async_track_time_interval(
                self.hass,
                self._async_periodic_auto_calibration,
                timedelta(hours=1),
            )
        )

        self.options = {}

        # TODO: This is only here because we haven't set up migration of config
        # entries yet, so some users might not have this defined after an update.
        self.options[CONF_ATTENUATION] = DEFAULT_ATTENUATION
        self.options[CONF_DEVTRACK_TIMEOUT] = DEFAULT_DEVTRACK_TIMEOUT
        self.options[CONF_MAX_RADIUS] = DEFAULT_MAX_RADIUS
        self.options[CONF_MAX_VELOCITY] = DEFAULT_MAX_VELOCITY
        self.options[CONF_REF_POWER] = DEFAULT_REF_POWER
        self.options[CONF_SMOOTHING_SAMPLES] = DEFAULT_SMOOTHING_SAMPLES
        self.options[CONF_UPDATE_INTERVAL] = DEFAULT_UPDATE_INTERVAL
        self.options[CONF_RSSI_OFFSETS] = {}

        if hasattr(entry, "options"):
            # Firstly, on some calls (specifically during reload after settings changes)
            # we seem to get called with a non-existant config_entry.
            # Anyway... if we DO have one, convert it to a plain dict so we can
            # serialise it properly when it goes into the device and scanner classes.
            for key, val in entry.options.items():
                if key in (
                    CONF_ATTENUATION,
                    CONF_DEVICES,
                    CONF_DEVTRACK_TIMEOUT,
                    CONF_MAX_RADIUS,
                    CONF_MAX_VELOCITY,
                    CONF_REF_POWER,
                    CONF_SMOOTHING_SAMPLES,
                    CONF_RSSI_OFFSETS,
                    CONFDATA_SCANNER_POSITIONS,
                    CONFDATA_FLOORS,
                    CONFDATA_ROOMS,
                ):
                    self.options[key] = val

        self.devices: dict[str, BermudaDevice] = {}
        # self.updaters: dict[str, BermudaPBDUCoordinator] = {}

        # Register the dump_devices service
        hass.services.async_register(
            DOMAIN,
            "dump_devices",
            self.service_dump_devices,
            vol.Schema(
                {
                    vol.Optional("addresses"): cv.string,
                    vol.Optional("configured_devices"): cv.boolean,
                    vol.Optional("redact"): cv.boolean,
                }
            ),
            SupportsResponse.ONLY,
        )

        # Register the check_scanner_broadcasting service
        hass.services.async_register(
            DOMAIN,
            "check_scanner_broadcasting",
            self.service_check_scanner_broadcasting,
            vol.Schema({}),
            SupportsResponse.ONLY,
        )

        # Register for newly discovered / changed BLE devices
        if self.config_entry is not None:
            self.config_entry.async_on_unload(
                bluetooth.async_register_callback(
                    self.hass,
                    self.async_handle_advert,
                    bluetooth.BluetoothCallbackMatcher(connectable=False),
                    bluetooth.BluetoothScanningMode.ACTIVE,
                )
            )

    @property
    def scanner_list(self):
        return self._scanner_list

    @property
    def get_scanners(self) -> set[BermudaDevice]:
        return self._scanners

    def init_floors(self) -> bool:
        """Check if the system has floors configured, and enable sensors."""
        _have_floors: bool = False
        for area in self.ar.async_list_areas():
            if area.floor_id is not None:
                _have_floors = True
                break
        _LOGGER.debug("Have_floors is %s", _have_floors)
        return _have_floors

    def scanner_list_add(self, scanner_device: BermudaDevice):
        self._scanner_list.add(scanner_device.address)
        self._scanners.add(scanner_device)
        async_dispatcher_send(self.hass, SIGNAL_SCANNERS_CHANGED)

    def scanner_list_del(self, scanner_device: BermudaDevice):
        self._scanner_list.remove(scanner_device.address)
        self._scanners.remove(scanner_device)
        async_dispatcher_send(self.hass, SIGNAL_SCANNERS_CHANGED)

    def get_manufacturer_from_id(self, uuid: int | str) -> tuple[str, bool] | tuple[None, None]:
        """
        An opinionated Bluetooth UUID to Name mapper.

        - uuid must be four hex chars in a string, or an `int`

        Retreives the manufacturer name from the Bluetooth SIG Member UUID listing,
        using a cached copy of https://bitbucket.org/bluetooth-SIG/public/src/main/assigned_numbers/uuids/member_uuids.yaml

        HOWEVER: Bermuda adds some opinionated overrides for the benefit of user clarity:
        - Legal entity names may be overriden with well-known brand names
        - Special-use prefixes may be tagged as such (eg iBeacon etc)
        - Generics can be excluded by setting exclude_generics=True
        """
        if isinstance(uuid, str):
            uuid = int(uuid.replace(":", ""), 16)

        _generic = False
        # Because iBeacon and (soon) GFMD and AppleFindmy etc are common protocols, they
        # don't do a good job of uniquely identifying a manufacturer, so we use them
        # as fallbacks only.
        if uuid == 0x0BA9:
            # allterco robotics, aka...
            _name = "Shelly Devices"
        elif uuid == 0x004C:
            # Apple have *many* UUIDs, but since they don't OEM for others (AFAIK)
            # and only the iBeacon / FindMy adverts seem to be third-partied, match just
            # this one instead of their entire set.
            _name = "Apple Inc."
            _generic = True
        elif uuid == 0x181C:
            _name = "BTHome v1 cleartext"
            _generic = True
        elif uuid == 0x181E:
            _name = "BTHome v1 encrypted"
            _generic = True
        elif uuid == 0xFCD2:
            _name = "BTHome V2"  # Sponsored by Allterco / Shelly
            _generic = True
        elif uuid in self.member_uuids:
            _name = self.member_uuids[uuid]
            # Hardware manufacturers who OEM MAC PHYs etc, or offer the use
            # of their OUIs to third parties (specific known ones can be moved
            # to a case in the above conditions).
            if any(x in _name for x in ["Google", "Realtek"]):
                _generic = True
        elif uuid in self.company_uuids:
            _name = self.company_uuids[uuid]
            _generic = False
        else:
            return (None, None)
        return (_name, _generic)

    async def async_load_manufacturer_ids(self):
        """Import yaml files containing manufacturer name mappings."""
        try:
            # https://bitbucket.org/bluetooth-SIG/public/src/main/assigned_numbers/uuids/member_uuids.yaml
            file_path = self.hass.config.path(
                f"custom_components/{DOMAIN}/manufacturer_identification/member_uuids.yaml"
            )
            async with aiofiles.open(file_path) as f:
                mi_yaml = yaml.safe_load(await f.read())["uuids"]
            self.member_uuids: dict[int, str] = {member["uuid"]: member["name"] for member in mi_yaml}

            # https://bitbucket.org/bluetooth-SIG/public/src/main/assigned_numbers/company_identifiers/company_identifiers.yaml
            file_path = self.hass.config.path(
                f"custom_components/{DOMAIN}/manufacturer_identification/company_identifiers.yaml"
            )
            async with aiofiles.open(file_path) as f:
                ci_yaml = yaml.safe_load(await f.read())["company_identifiers"]
            self.company_uuids: dict[int, str] = {member["value"]: member["name"] for member in ci_yaml}
        finally:
            # Ensure that an issue reading these files (which are optional, really) doesn't stop the whole show.
            self._waitingfor_load_manufacturer_ids = False

    def load_scanner_positions(self):
        """Load scanner positions from bermuda.yaml or config entry for trilateration.

        Loading priority:
        1. bermuda.yaml in HA config dir (ESPresense-compatible format)
        2. Config entry data (JSON import from config flow)
        """
        _LOGGER.info("=== LOADING SCANNER POSITIONS ===")
        _LOGGER.debug("Current scanners: %s", [d.address for d in self.get_scanners])

        positions_loaded = 0
        self.map_floors = {}
        self.map_rooms = {}

        # Try loading ESPresense-compatible YAML config first
        yaml_loaded = self._load_yaml_config()
        if yaml_loaded:
            positions_loaded = yaml_loaded
        else:
            # Fall back to config entry data
            positions_loaded = self._load_config_entry_positions()

        if positions_loaded > 0:
            _LOGGER.info("Loaded %d scanner positions for position tracking", positions_loaded)
        else:
            _LOGGER.warning("No valid scanner positions loaded")
            _LOGGER.warning("  Create bermuda.yaml in your HA config dir or use Bulk Import")

        if self.map_rooms:
            _LOGGER.info("Loaded %d rooms for zone detection", len(self.map_rooms))
        if self.map_floors:
            _LOGGER.info("Loaded %d floors", len(self.map_floors))

    def _load_yaml_config(self) -> int:
        """Load ESPresense-compatible YAML config from bermuda.yaml.

        Returns number of scanner positions loaded, or 0 if file not found.
        """
        import os

        yaml_path = self.hass.config.path(CONF_YAML_CONFIG_FILE)
        if not os.path.isfile(yaml_path):
            _LOGGER.debug("No bermuda.yaml found at %s", yaml_path)
            return 0

        _LOGGER.info("Loading ESPresense-compatible config from %s", yaml_path)

        try:
            with open(yaml_path) as f:
                config = yaml.safe_load(f)
        except Exception as e:
            _LOGGER.error("Failed to parse bermuda.yaml: %s", e)
            return 0

        if not isinstance(config, dict):
            _LOGGER.error("bermuda.yaml: invalid format (expected dict)")
            return 0

        positions_loaded = 0

        # Load Kalman filter settings from YAML
        filtering = config.get("filtering", {})
        if filtering:
            from .const import CONF_KALMAN_PROCESS_NOISE
            from .const import CONF_KALMAN_MEASUREMENT_NOISE
            from .const import CONF_KALMAN_MAX_VELOCITY
            # Store as instance attributes for use by position pipeline
            self._yaml_kalman_process_noise = filtering.get("process_noise", 0.01)
            self._yaml_kalman_measurement_noise = filtering.get("measurement_noise", 0.1)
            self._yaml_kalman_max_velocity = filtering.get("max_velocity", 0.5)
            _LOGGER.info(
                "Kalman filter settings: process_noise=%.3f, measurement_noise=%.3f, max_velocity=%.2f",
                self._yaml_kalman_process_noise,
                self._yaml_kalman_measurement_noise,
                self._yaml_kalman_max_velocity,
            )

        # Load floors with nested rooms (ESPresense format)
        yaml_floors = config.get("floors", [])
        for floor_data in yaml_floors:
            floor_id = floor_data.get("id") or _slugify_name(floor_data.get("name", ""))
            if not floor_id:
                continue

            floor_entry = {
                "id": floor_id,
                "name": floor_data.get("name", floor_id),
                "bounds": floor_data.get("bounds", []),
                "rooms": [],
            }

            # ESPresense nests rooms inside floors
            for room_data in floor_data.get("rooms", []):
                room_name = room_data.get("name", "")
                room_id = room_data.get("id") or _slugify_name(room_name)
                if not room_id:
                    continue

                room_entry = {
                    "id": room_id,
                    "name": room_name,
                    "floor": floor_id,
                    "points": room_data.get("points", []),
                }
                floor_entry["rooms"].append(room_entry)
                self.map_rooms[room_id] = room_entry

            self.map_floors[floor_id] = floor_entry

        # Load nodes (scanner positions)
        yaml_nodes = config.get("nodes", [])
        for node in yaml_nodes:
            node_name = node.get("name", "")
            node_id = node.get("id") or node_name
            point = node.get("point")
            node_floors = node.get("floors", [])

            if not point or len(point) < 3:
                _LOGGER.warning("bermuda.yaml: Node '%s' missing point [x,y,z]", node_id)
                continue

            # Match node to a bermuda scanner by name or address
            matched_scanner = self._match_yaml_node_to_scanner(node_id, node_name)
            if matched_scanner:
                matched_scanner.position = tuple(point)
                matched_scanner.node_floors = node_floors if node_floors else None
                positions_loaded += 1
                _LOGGER.info(
                    "  Loaded position for scanner %s (%s): (%.2f, %.2f, %.2f) floors=%s",
                    node_name,
                    matched_scanner.address,
                    point[0], point[1], point[2],
                    node_floors,
                )
            else:
                _LOGGER.warning(
                    "  bermuda.yaml: Node '%s' not matched to any known scanner",
                    node_id,
                )

        # Load timeout settings
        timeout = config.get("timeout")
        if timeout is not None:
            _LOGGER.debug("bermuda.yaml: timeout=%d", timeout)

        return positions_loaded

    def _match_yaml_node_to_scanner(self, node_id: str, node_name: str) -> BermudaDevice | None:
        """Match a YAML node definition to a bermuda scanner device.

        Tries matching by:
        1. Node ID as MAC address
        2. Node name matching scanner name (case-insensitive)
        3. Node name contained in scanner name
        """
        # Try node_id as MAC address
        try:
            normalized = mac_norm(node_id)
            if normalized in self.devices and self.devices[normalized].is_scanner:
                return self.devices[normalized]
        except (ValueError, TypeError):
            pass

        # Try matching by name
        name_lower = node_name.lower() if node_name else ""
        id_lower = node_id.lower() if node_id else ""

        for scanner in self._scanners:
            scanner_name = (scanner.name or "").lower()
            scanner_name_bt = (scanner.name_bt_local_name or "").lower()
            scanner_name_user = (scanner.name_by_user or "").lower()

            # Exact name match
            if name_lower and (
                scanner_name == name_lower
                or scanner_name_bt == name_lower
                or scanner_name_user == name_lower
            ):
                return scanner

            # ID match
            if id_lower and (
                scanner_name == id_lower
                or scanner_name_bt == id_lower
                or scanner_name_user == id_lower
            ):
                return scanner

            # Partial name match (node name contained in scanner name)
            if name_lower and len(name_lower) >= 3 and (
                name_lower in scanner_name
                or name_lower in scanner_name_bt
                or name_lower in scanner_name_user
            ):
                return scanner

        return None

    def _load_config_entry_positions(self) -> int:
        """Load scanner positions from config entry data (JSON import format)."""
        positions_loaded = 0

        config_floors = self.options.get(CONFDATA_FLOORS, [])
        config_rooms = self.options.get(CONFDATA_ROOMS, [])
        config_positions = self.options.get(CONFDATA_SCANNER_POSITIONS, {})

        # Load floors
        for floor in config_floors:
            self.map_floors[floor["id"]] = floor

        # Load rooms
        for room in config_rooms:
            self.map_rooms[room["id"]] = room

        # Apply scanner positions
        for m, pos_data in config_positions.items():
            mac = mac_norm(m)
            if mac in self.devices and self.devices[mac].is_scanner:
                self.devices[mac].position = tuple(pos_data["point"])
                # Load floor assignments if present
                if "floors" in pos_data:
                    self.devices[mac].node_floors = pos_data["floors"]
                positions_loaded += 1
                _LOGGER.info(
                    "  Loaded position for scanner %s (%s): (%.2f, %.2f, %.2f)",
                    pos_data.get("name", mac),
                    mac,
                    pos_data["point"][0], pos_data["point"][1], pos_data["point"][2],
                )

        return positions_loaded

    @callback
    def handle_devreg_changes(self, ev: Event[EventDeviceRegistryUpdatedData]):
        """
        Update our scanner list if the device registry is changed.

        This catches area changes (on scanners) and any new/changed
        Private BLE Devices.
        """
        if ev.data["action"] == "update":
            _LOGGER.debug("Device registry UPDATE. ev: %s changes: %s", ev, ev.data["changes"])
        else:
            _LOGGER.debug("Device registry has changed. ev: %s", ev)

        device_id = ev.data.get("device_id")

        if ev.data["action"] in {"create", "update"}:
            if device_id is None:
                _LOGGER.error("Received Device Registry create/update without a device_id. ev.data: %s", ev.data)
                return

            # First look for any of our devices that have a stored id on them, it'll be quicker.
            for device in self.devices.values():
                if device.entry_id == device_id:
                    # We matched, most likely a scanner.
                    if device.is_scanner:
                        self._refresh_scanners(force=True)
                        return
            # Didn't match an existing, work through the connections etc.

            # Pull up the device registry entry for the device_id
            if device_entry := self.dr.async_get(ev.data["device_id"]):
                # Work out if it's a device that interests us and respond appropriately.
                for conn_type, _conn_id in device_entry.connections:
                    if conn_type == "private_ble_device":
                        _LOGGER.debug("Trigger updating of Private BLE Devices")
                        self._do_private_device_init = True
                    elif conn_type == "ibeacon":
                        # this was probably us, nothing else to do
                        pass
                    else:
                        for ident_type, ident_id in device_entry.identifiers:
                            if ident_type == DOMAIN:
                                # One of our sensor devices!
                                try:
                                    if _device := self.devices[ident_id.lower()]:
                                        _device.name_by_user = device_entry.name_by_user
                                        _device.make_name()
                                except KeyError:
                                    pass
                        # might be a scanner, so let's refresh those
                        _LOGGER.debug("Trigger updating of Scanner Listings")
                        self._scanner_init_pending = True
            else:
                _LOGGER.error(
                    "Received DR update/create but device id does not exist: %s",
                    ev.data["device_id"],
                )

        elif ev.data["action"] == "remove":
            device_found = False
            for scanner in self.get_scanners:
                if scanner.entry_id == device_id:
                    _LOGGER.debug(
                        "Scanner %s removed, trigger update of scanners",
                        scanner.name,
                    )
                    self._scanner_init_pending = True
                    device_found = True
            if not device_found:
                # If we save the private ble device's device_id into devices[].entry_id
                # we could check ev.data["device_id"] against it to decide if we should
                # rescan PBLE devices. But right now we don't, so scan 'em anyway.
                _LOGGER.debug("Opportunistic trigger of update for Private BLE Devices")
                self._do_private_device_init = True
        # The co-ordinator will only get updates if we have created entities already.
        # Since this might not always be the case (say, private_ble_device loads after
        # we do), then we trigger an update here with the expectation that we got a
        # device registry update after the private ble device was created. There might
        # be other corner cases where we need to trigger our own update here, so test
        # carefully and completely if you are tempted to remove / alter this. Bermuda
        # will skip an update cycle if it detects one already in progress.
        # FIXME: self._async_update_data_internal()

    @callback
    def async_handle_advert(
        self,
        service_info: BluetoothServiceInfoBleak,
        change: BluetoothChange,
    ) -> None:
        """
        Handle an incoming advert callback from the bluetooth integration.

        These should come in as adverts are received, rather than on our update schedule.
        The data *should* be as fresh as can be, but actually the backend only sends
        these periodically (mainly when the data changes, I think). So it's no good for
        responding to changing rssi values, but it *is* good for seeding our updates in case
        there are no defined sensors yet (or the defined ones are away).
        """
        # _LOGGER.debug(
        #     "New Advert! change: %s, scanner: %s mac: %s name: %s serviceinfo: %s",
        #     change,
        #     service_info.source,
        #     service_info.address,
        #     service_info.name,
        #     service_info,
        # )

        # If there are no active entities created after Bermuda's
        # initial setup, then no updates will be triggered on the co-ordinator.
        # So let's check if we haven't updated recently, and do so...
        if self.stamp_last_update < monotonic_time_coarse() - (UPDATE_INTERVAL * 2):
            self._async_update_data_internal()

    def _check_all_platforms_created(self, address):
        """Checks if all platforms have finished loading a device's entities."""
        dev = self._get_device(address)
        if dev is not None:
            if all(
                [
                    dev.create_sensor_done,
                    dev.create_tracker_done,
                    dev.create_number_done,
                ]
            ):
                dev.create_all_done = True

    def sensor_created(self, address):
        """Allows sensor platform to report back that sensors have been set up."""
        dev = self._get_device(address)
        if dev is not None:
            dev.create_sensor_done = True
            # _LOGGER.debug("Sensor confirmed created for %s", address)
        else:
            _LOGGER.warning("Very odd, we got sensor_created for non-tracked device")
        self._check_all_platforms_created(address)

    def device_tracker_created(self, address):
        """Allows device_tracker platform to report back that sensors have been set up."""
        dev = self._get_device(address)
        if dev is not None:
            dev.create_tracker_done = True
            # _LOGGER.debug("Device_tracker confirmed created for %s", address)
        else:
            _LOGGER.warning("Very odd, we got sensor_created for non-tracked device")
        self._check_all_platforms_created(address)

    def number_created(self, address):
        """Receives report from number platform that sensors have been set up."""
        dev = self._get_device(address)
        if dev is not None:
            dev.create_number_done = True
        self._check_all_platforms_created(address)

    # def button_created(self, address):
    #     """Receives report from number platform that sensors have been set up."""
    #     dev = self._get_device(address)
    #     if dev is not None:
    #         dev.create_button_done = True
    #     self._check_all_platforms_created(address)

    def count_active_devices(self) -> int:
        """
        Returns the number of bluetooth devices that have recent timestamps.

        Useful as a general indicator of health
        """
        stamp = monotonic_time_coarse() - 10  # seconds
        fresh_count = 0
        for device in self.devices.values():
            if device.last_seen > stamp:
                fresh_count += 1
        return fresh_count

    def count_active_scanners(self, max_age=10) -> int:
        """Returns count of scanners that have recently sent updates."""
        stamp = monotonic_time_coarse() - max_age  # seconds
        fresh_count = 0
        for scanner in self.get_active_scanner_summary():
            if scanner.get("last_stamp", 0) > stamp:
                fresh_count += 1
        return fresh_count

    def get_active_scanner_summary(self) -> list[dict]:
        """
        Returns a list of dicts suitable for seeing which scanners
        are configured in the system and how long it has been since
        each has returned an advertisement.
        """
        stamp = monotonic_time_coarse()
        return [
            {
                "name": scannerdev.name,
                "address": scannerdev.address,
                "last_stamp": scannerdev.last_seen,
                "last_stamp_age": stamp - scannerdev.last_seen,
            }
            for scannerdev in self.get_scanners
        ]

    def collect_calibration_rssi_data(
        self,
        target_device_address: str,
        min_samples: int = 10,
    ) -> dict[str, list[float]]:
        """
        Collect RSSI sample data from all scanners for a specific calibration target device.

        Returns a dict mapping scanner_address -> list of RSSI samples.
        Only includes scanners with at least min_samples recent RSSI readings.

        Args:
            target_device_address: MAC address of the calibration beacon device
            min_samples: Minimum number of samples required per scanner (default 10)

        Returns:
            Dict mapping scanner_address -> list of RSSI values (floats)
        """
        target_device = self._get_device(target_device_address)
        if target_device is None:
            _LOGGER.warning(
                "Calibration target device %s not found in devices",
                target_device_address,
            )
            return {}

        scanner_rssi_data: dict[str, list[float]] = {}

        # Iterate through all adverts for this device
        for advert_key, advert in target_device.adverts.items():
            scanner_addr = advert.scanner_address

            # Only use scanners (not device-to-device adverts)
            if scanner_addr not in self.devices:
                continue

            scanner_device = self.devices[scanner_addr]
            if not scanner_device.is_scanner:
                continue

            # Collect RSSI history from the advert object
            # hist_rssi is a deque with most recent values
            rssi_samples = list(advert.hist_rssi)

            # Filter out None values and ensure we have enough samples
            rssi_samples = [float(r) for r in rssi_samples if r is not None]

            if len(rssi_samples) >= min_samples:
                scanner_rssi_data[scanner_addr] = rssi_samples

        return scanner_rssi_data

    def get_scanner_positions_dict(self) -> dict[str, tuple[float, float, float]]:
        """
        Get scanner positions as a dict mapping scanner_address -> (x, y, z).

        Returns:
            Dict of scanner positions, empty dict if no positions configured
        """
        scanner_positions: dict[str, tuple[float, float, float]] = {}

        for scanner_addr, scanner_device in self.devices.items():
            if scanner_device.is_scanner and scanner_device.position is not None:
                scanner_positions[scanner_addr] = scanner_device.position

        return scanner_positions

    def collect_scanner_to_scanner_rssi_data(
        self,
        min_samples: int = 10,
    ) -> dict[tuple[str, str], list[float]]:
        """
        Collect RSSI measurements between all scanner pairs for auto-calibration.

        Returns dict mapping (scanner_A_address, scanner_B_address) -> list of RSSI samples.
        Only includes pairs where scanner A has received broadcasts from scanner B.

        This enables scanner-to-scanner RF ranging calibration without needing an external beacon.
        Requires that scanners are configured to broadcast BLE advertisements.

        Args:
            min_samples: Minimum number of samples required per scanner pair (default 10)

        Returns:
            Dict mapping (scanner_A_addr, scanner_B_addr) -> list of RSSI values (floats)
            Empty dict if insufficient data
        """
        scanner_pairs_rssi: dict[tuple[str, str], list[float]] = {}

        # Get all scanner devices
        scanner_devices = {addr: dev for addr, dev in self.devices.items() if dev.is_scanner}

        if len(scanner_devices) < 2:
            _LOGGER.debug("Scanner-to-scanner calibration: Need at least 2 scanners, found %d", len(scanner_devices))
            return {}
        
        _LOGGER.debug(
            "Scanner-to-scanner calibration: Found %d scanners: %s",
            len(scanner_devices),
            ", ".join(scanner_devices.keys()),
        )

        # For each device in the system, check if it's a scanner broadcasting to other scanners
        for device_addr, device in self.devices.items():
            # Check if this device is a scanner (i.e., it broadcasts BLE advertisements)
            # that other scanners have received
            
            # Look at all the adverts for this device
            # Each advert represents: "device_addr was seen by scanner_addr"
            # The advert_key tuple is (device_address, scanner_address)
            for advert_key, advert in device.adverts.items():
                # advert_key is always (device_address, scanner_address)
                # device_address should match this device, scanner_address is who received it
                if isinstance(advert_key, tuple) and len(advert_key) == 2:
                    transmitting_device, receiving_scanner = advert_key
                    
                    # Verify this is an advert for the current device
                    if transmitting_device != device_addr:
                        continue
                    
                    # Check if BOTH the transmitting device AND receiving scanner are scanners
                    # This means we have scanner-to-scanner communication
                    if transmitting_device in scanner_devices and receiving_scanner in scanner_devices:
                        # Found scanner-to-scanner pair!
                        # receiving_scanner saw transmitting_device
                        
                        rssi_samples = list(advert.hist_rssi)
                        rssi_samples = [float(r) for r in rssi_samples if r is not None]
                        
                        if len(rssi_samples) >= min_samples:
                            # Store as (receiver, transmitter) tuple
                            pair_key = (receiving_scanner, transmitting_device)
                            scanner_pairs_rssi[pair_key] = rssi_samples
                            _LOGGER.debug(
                                "Scanner-to-scanner: Found %d RSSI samples from %s received by %s",
                                len(rssi_samples),
                                transmitting_device,
                                receiving_scanner,
                            )
                        elif len(rssi_samples) > 0:
                            # Has some samples but not enough
                            _LOGGER.debug(
                                "Scanner-to-scanner: Insufficient samples from %s received by %s (%d < %d needed)",
                                transmitting_device,
                                receiving_scanner,
                                len(rssi_samples),
                                min_samples,
                            )

        if scanner_pairs_rssi:
            _LOGGER.info(
                "Scanner-to-scanner calibration: Found %d scanner pairs with sufficient data (min %d samples)",
                len(scanner_pairs_rssi),
                min_samples,
            )
            # Log summary of which scanners are broadcasting
            broadcasting_scanners = set()
            for receiver, transmitter in scanner_pairs_rssi.keys():
                broadcasting_scanners.add(transmitter)
            _LOGGER.info(
                "Scanner-to-scanner: Detected %d broadcasting scanners: %s",
                len(broadcasting_scanners),
                ", ".join(sorted(broadcasting_scanners)),
            )
        else:
            _LOGGER.warning(
                "Scanner-to-scanner calibration: No scanner pairs found with sufficient RSSI data (need %d samples). "
                "Ensure scanners are configured to broadcast BLE advertisements.",
                min_samples,
            )

        return scanner_pairs_rssi

    async def _async_periodic_auto_calibration(self, now=None):
        """
        Periodic task to run automatic scanner-to-scanner calibration.

        Runs every hour, checks if calibration is enabled and interval has elapsed.
        Uses scanner-to-scanner RF ranging method only (beacon method not supported for background).
        """
        # Check if auto-calibration is enabled
        if not self.options.get(CONF_AUTO_CALIBRATION_ENABLED, DEFAULT_AUTO_CALIBRATION_ENABLED):
            return

        # Check if already running
        if self._calibration_in_progress:
            _LOGGER.debug("Auto-calibration already in progress, skipping this cycle")
            return

        # Check if enough time has elapsed since last calibration
        interval_hours = self.options.get(
            CONF_AUTO_CALIBRATION_INTERVAL_HOURS,
            DEFAULT_AUTO_CALIBRATION_INTERVAL_HOURS,
        )
        current_time = monotonic_time_coarse()
        elapsed_hours = (current_time - self._last_calibration_time) / 3600

        if elapsed_hours < interval_hours:
            _LOGGER.debug(
                "Auto-calibration: %.1f hours since last run, need %.1f hours",
                elapsed_hours,
                interval_hours,
            )
            return

        _LOGGER.info("Starting automatic RSSI calibration (scanner-to-scanner method)")
        self._calibration_in_progress = True

        try:
            await self._run_scanner_to_scanner_calibration()
        except Exception as e:  # noqa: BLE001
            _LOGGER.exception("Auto-calibration failed with exception: %s", e)
            await async_create_notification(
                self.hass,
                f"Automatic RSSI calibration failed: {e}",
                title="Bermuda Auto-Calibration Error",
                notification_id=NOTIFICATION_ID_CALIBRATION_INSUFFICIENT,
            )
        finally:
            self._calibration_in_progress = False
            self._last_calibration_time = current_time

    async def _run_scanner_to_scanner_calibration(self):
        """
        Run scanner-to-scanner RF ranging calibration and apply results.

        Collects RSSI data between scanner pairs, calculates offsets,
        and updates config entry with new offsets.

        Creates persistent notifications for success/failure.
        """
        from .util import calculate_scanner_offsets_from_scanner_pairs
        from .util import get_scanner_pair_quality_metrics

        # Get configuration
        min_samples = self.options.get(CONF_AUTO_CALIBRATION_SAMPLES, 50)
        ref_power = self.options.get(CONF_REF_POWER, -55.0)
        attenuation = self.options.get(CONF_ATTENUATION, 3.5)

        # Collect scanner-to-scanner RSSI data
        scanner_pairs_rssi = self.collect_scanner_to_scanner_rssi_data(min_samples=min_samples)

        # Check for insufficient data
        if len(scanner_pairs_rssi) < AUTO_CALIBRATION_MIN_SCANNER_PAIRS:
            # Check notification cooldown
            current_time = monotonic_time_coarse()
            if current_time - self._last_calibration_notification > CALIBRATION_NOTIFICATION_COOLDOWN:
                # Detect non-broadcasting scanners
                non_broadcasting = await self._detect_non_broadcasting_scanners()

                if non_broadcasting:
                    # Create notification about non-broadcasting scanners
                    scanner_names = ", ".join([dev.name for dev in non_broadcasting[:5]])
                    if len(non_broadcasting) > 5:
                        scanner_names += f" (+{len(non_broadcasting) - 5} more)"

                    message = (
                        f"**Automatic RSSI calibration requires broadcasting scanners**\n\n"
                        f"Found {len(scanner_pairs_rssi)} scanner pairs with data, "
                        f"need {AUTO_CALIBRATION_MIN_SCANNER_PAIRS}.\n\n"
                        f"**Scanners not broadcasting:** {scanner_names}\n\n"
                        f"Configure ESPHome proxies to broadcast BLE advertisements:\n"
                        f"```yaml\n"
                        f"esp32_ble_tracker:\n"
                        f"  scan_parameters:\n"
                        f"    active: true\n"
                        f"  # Enable BLE broadcasting\n"
                        f"  on_ble_advertise:\n"
                        f"    - then:\n"
                        f"        - ble_advertise.start:\n"
                        f"            transmit_power: 3dBm\n"
                        f"```\n\n"
                        f"Alternatively, disable auto-calibration or use manual beacon calibration."
                    )
                    await async_create_notification(
                        self.hass,
                        message,
                        title="Bermuda Auto-Calibration: No Broadcasting Scanners",
                        notification_id=NOTIFICATION_ID_CALIBRATION_NO_BROADCASTING,
                    )
                else:
                    # Generic insufficient data message
                    message = (
                        f"**Automatic RSSI calibration: Insufficient data**\n\n"
                        f"Found {len(scanner_pairs_rssi)} scanner pairs with data, "
                        f"need {AUTO_CALIBRATION_MIN_SCANNER_PAIRS}.\n\n"
                        f"Calibration will retry in {self.options.get(CONF_AUTO_CALIBRATION_INTERVAL_HOURS, 24)} hours. "
                        f"Ensure scanners are configured to broadcast BLE advertisements and have been running "
                        f"for sufficient time to collect {min_samples} samples."
                    )
                    await async_create_notification(
                        self.hass,
                        message,
                        title="Bermuda Auto-Calibration: Insufficient Data",
                        notification_id=NOTIFICATION_ID_CALIBRATION_INSUFFICIENT,
                    )

                self._last_calibration_notification = current_time

            _LOGGER.warning(
                "Auto-calibration: Insufficient scanner pairs (%d < %d)",
                len(scanner_pairs_rssi),
                AUTO_CALIBRATION_MIN_SCANNER_PAIRS,
            )
            return

        # Get scanner positions
        scanner_positions = self.get_scanner_positions_dict()

        # Calculate offsets
        new_offsets = calculate_scanner_offsets_from_scanner_pairs(
            scanner_pairs_rssi,
            scanner_positions,
            ref_power,
            attenuation,
        )

        if not new_offsets:
            _LOGGER.error("Auto-calibration: Failed to calculate offsets")
            return

        # Calculate quality metrics for reporting
        metrics = get_scanner_pair_quality_metrics(scanner_pairs_rssi, scanner_positions)

        # Build success notification with metrics
        status_lines = [
            "**Automatic RSSI Calibration Successful**\n",
            f"Calibrated {len(new_offsets)} scanners using {len(scanner_pairs_rssi)} scanner pairs.\n",
            "\n**Calculated Offsets:**\n",
        ]

        for scanner_addr, offset in sorted(new_offsets.items(), key=lambda x: x[1], reverse=True):
            scanner_name = self.devices[scanner_addr].name if scanner_addr in self.devices else scanner_addr
            status_lines.append(f"- {scanner_name}: {offset:+.1f} dBm")

        status_lines.append("\n**Scanner Pair Quality:**\n")
        status_lines.append("|Receiver → Transmitter|Distance|Samples|RSSI Median|RSSI StdDev|")
        status_lines.append("|---|---:|---:|---:|---:|")

        for (receiver_addr, transmitter_addr), metric in list(metrics.items())[:10]:  # Limit to 10 pairs
            receiver_name = self.devices[receiver_addr].name
            transmitter_name = self.devices[transmitter_addr].name
            status_lines.append(
                f"|{receiver_name} → {transmitter_name}"
                f"|{metric['distance']:.2f}m"
                f"|{metric['sample_count']}"
                f"|{metric['rssi_median']:.1f} dBm"
                f"|{metric['rssi_std']:.1f} dBm|"
            )

        if len(metrics) > 10:
            status_lines.append(f"\n*... and {len(metrics) - 10} more pairs*")

        # Dismiss any previous error notifications
        await async_dismiss_notification(self.hass, NOTIFICATION_ID_CALIBRATION_INSUFFICIENT)
        await async_dismiss_notification(self.hass, NOTIFICATION_ID_CALIBRATION_NO_BROADCASTING)

        # Create success notification (auto-dismiss after 1 hour)
        await async_create_notification(
            self.hass,
            "\n".join(status_lines),
            title="Bermuda Auto-Calibration Complete",
            notification_id=NOTIFICATION_ID_CALIBRATION_SUCCESS,
        )

        # Update config entry with new offsets
        self.options[CONF_RSSI_OFFSETS] = new_offsets
        self.hass.config_entries.async_update_entry(
            self.config_entry,
            options=self.options,
        )

        _LOGGER.info(
            "Auto-calibration complete: %d scanners calibrated from %d pairs",
            len(new_offsets),
            len(scanner_pairs_rssi),
        )

    async def _detect_non_broadcasting_scanners(self) -> list[BermudaDevice]:
        """
        Detect scanners that are not broadcasting BLE advertisements.

        A scanner is considered non-broadcasting if:
        - It is marked as a scanner (is_scanner=True)
        - No other scanners have received advertisements from it

        Returns:
            List of BermudaDevice objects for non-broadcasting scanners
        """
        scanner_devices = {addr: dev for addr, dev in self.devices.items() if dev.is_scanner}

        if len(scanner_devices) < 2:
            _LOGGER.debug("Cannot detect broadcasting with < 2 scanners")
            return list(scanner_devices.values())  # All scanners if only 1

        non_broadcasting = []
        broadcasting_info = {}  # Track which scanners see which

        for scanner_addr, scanner_device in scanner_devices.items():
            # Check if any OTHER scanner has received from this scanner
            is_broadcasting = False
            receivers = []

            # Iterate through all devices to find this scanner in their adverts
            for device_addr, device in self.devices.items():
                # Skip if checking against self
                if device_addr == scanner_addr:
                    continue
                
                # Check if this device has adverts from our scanner
                for advert_key in device.adverts.keys():
                    if isinstance(advert_key, tuple) and len(advert_key) == 2:
                        transmitting_device, receiving_scanner = advert_key
                        
                        # Found: scanner_addr transmitted to this device
                        if transmitting_device == scanner_addr and receiving_scanner in scanner_devices:
                            is_broadcasting = True
                            receivers.append(receiving_scanner)

            if is_broadcasting:
                broadcasting_info[scanner_addr] = receivers
                _LOGGER.debug(
                    "Scanner %s IS broadcasting (seen by %d scanners: %s)",
                    scanner_addr,
                    len(receivers),
                    ", ".join(receivers[:3]) + ("..." if len(receivers) > 3 else ""),
                )
            else:
                non_broadcasting.append(scanner_device)
                _LOGGER.debug(
                    "Scanner %s is NOT broadcasting (not seen by any other scanner)",
                    scanner_addr,
                )

        # Log summary
        _LOGGER.info(
            "Broadcasting detection: %d/%d scanners broadcasting, %d not broadcasting",
            len(broadcasting_info),
            len(scanner_devices),
            len(non_broadcasting),
        )

        return non_broadcasting

    def _get_device(self, address: str) -> BermudaDevice | None:
        """Search for a device entry based on mac address."""
        # mac_norm tries to return a lower-cased, colon-separated mac address.
        # failing that, it returns the original, lower-cased.
        try:
            return self.devices[mac_norm(address)]
        except KeyError:
            return None

    def _get_or_create_device(self, address: str) -> BermudaDevice:
        mac = mac_norm(address)
        try:
            return self.devices[mac]
        except KeyError:
            self.devices[mac] = device = BermudaDevice(mac, self)
            return device

    async def _async_update_data(self):
        """Implementation of DataUpdateCoordinator update_data function."""
        # return False
        self._async_update_data_internal()

    def _async_update_data_internal(self):
        """
        The primary update loop that processes almost all data in Bermuda.

        This works only with local data, so should be cheap to run
        (no network requests made etc). This function takes care of:

        - gathering all bluetooth adverts since last run and saving them into
          Bermuda's device objects
        - Updating all metadata
        - Performing rssi and statistical calculations
        - Making area determinations
        - (periodically) pruning device entries

        """
        if self._waitingfor_load_manufacturer_ids:
            _LOGGER.debug("Waiting for BT data load...")
            return True
        if self.update_in_progress:
            # Eeep!
            _LOGGER_SPAM_LESS.warning("update_still_running", "Previous update still running, skipping this cycle.")
            return False
        self.update_in_progress = True

        try:  # so we can still clean up update_in_progress
            nowstamp = monotonic_time_coarse()

            # The main "get all adverts from the backend" part.
            result_gather_adverts = self._async_gather_advert_data()

            self.update_metadevices()

            # Calculate per-device data
            #
            # Scanner entries have been loaded up with latest data, now we can
            # process data for all devices over all scanners.
            for device in self.devices.values():
                # Recalculate smoothed distances, last_seen etc
                device.calculate_data()

                if device.create_sensor and self.options.get(CONF_ENABLE_TRILATERATION, True):
                    # Calculate trilateration position if enabled and device is tracked
                    _LOGGER.debug(
                        "Device %s: create_sensor=%s, trilat_enabled=%s",
                        device.name,
                        device.create_sensor,
                        self.options.get(CONF_ENABLE_TRILATERATION, True),
                    )
                    min_scanners = self.options.get(CONF_MIN_TRILATERATION_SCANNERS, 2)

                    # Count scanners with valid positions and distances
                    _LOGGER.debug(
                        "Counting valid scanners for device %s (has %d adverts)",
                        device.name,
                        len(device.adverts),
                    )

                    # Validate scanners using shared helper (includes staleness check)
                    max_scanners = self.options.get(
                        CONF_MAX_TRILATERATION_SCANNERS,
                        DEFAULT_MAX_TRILATERATION_SCANNERS,
                    )
                    valid_scanners = validate_scanners_for_trilateration(
                        device,
                        nowstamp,
                        TRILATERATION_POSITION_TIMEOUT,
                        max_scanners=max_scanners,
                    )
                    scanner_count = len(valid_scanners)

                    if scanner_count >= min_scanners:
                        _LOGGER.debug(
                            "Device %s: Calling trilateration (scanners: %d >= %d)",
                            device.name,
                            scanner_count,
                            min_scanners,
                        )
                        debug_enabled = self.config_entry.options.get(
                            CONF_TRILATERATION_DEBUG, DEFAULT_TRILATERATION_DEBUG
                        )
                        result = calculate_position(device, nowstamp, debug_enabled)
                        if result:
                            # Apply Kalman filter to smooth position
                            if device._kalman_location is None:
                                settings = KalmanFilterSettings(
                                    process_noise=self.options.get(
                                        CONF_KALMAN_PROCESS_NOISE, DEFAULT_KALMAN_PROCESS_NOISE
                                    ),
                                    measurement_noise=self.options.get(
                                        CONF_KALMAN_MEASUREMENT_NOISE, DEFAULT_KALMAN_MEASUREMENT_NOISE
                                    ),
                                    max_velocity=self.options.get(
                                        CONF_KALMAN_MAX_VELOCITY, DEFAULT_KALMAN_MAX_VELOCITY
                                    ),
                                )
                                device._kalman_location = KalmanLocation(settings)

                            filtered_x, filtered_y, filtered_z = device._kalman_location.update(
                                result.x, result.y, result.z, nowstamp
                            )

                            _LOGGER.info(
                                "Position for %s: raw=(%.2f,%.2f,%.2f) filtered=(%.2f,%.2f,%.2f) "
                                "confidence=%.1f%% method=%s error=%.3f corr=%.3f",
                                device.name,
                                result.x, result.y, result.z,
                                filtered_x, filtered_y, filtered_z,
                                result.confidence,
                                result.method,
                                result.error,
                                result.correlation,
                            )

                            device.calculated_position = (filtered_x, filtered_y, filtered_z)
                            device.position_confidence = result.confidence
                            device.position_timestamp = nowstamp
                            device.position_method = result.method
                            device.position_error = result.error
                            device.position_correlation = result.correlation
                            device.position_room_id = result.room_id
                            device.position_floor_id = result.floor_id

                            # Determine room from position and override area assignment if enabled
                            if (
                                self.map_rooms
                                and self.options.get(CONF_TRILATERATION_OVERRIDE_AREA, True)
                                and result.confidence
                                >= self.options.get(CONF_TRILATERATION_AREA_MIN_CONFIDENCE, 30.0)
                            ):
                                # Use room_id from trilateration if available, else look up from position
                                room_area_id = result.room_id
                                if not room_area_id:
                                    room_area_id = find_room_for_position(
                                        (filtered_x, filtered_y, filtered_z),
                                        list(self.map_rooms.values()),
                                        list(self.map_floors.values()) if self.map_floors else None,
                                    )

                                if room_area_id:
                                    # Map to HA Area
                                    area = self.ar.async_get_area(room_area_id)
                                    if area:
                                        # Override the distance-based area assignment
                                        old_area = device.area_name
                                        device._update_area_and_floor(area.id)
                                        _LOGGER.info(
                                            "Device %s area: %s → %s via trilateration (confidence: %.1f%%)",
                                            device.name,
                                            old_area or "None",
                                            area.name,
                                            result.confidence,
                                        )
                                    else:
                                        _LOGGER.warning(
                                            "Room area_id '%s' not found in Home Assistant Areas",
                                            room_area_id,
                                        )
                                else:
                                    _LOGGER.debug(
                                        "Device %s position not in any defined room",
                                        device.name,
                                    )
                    else:
                        _LOGGER.debug(
                            "Device %s: Trilateration skipped - insufficient scanners (%d < %d)",
                            device.name,
                            scanner_count,
                            min_scanners,
                        )
                elif not device.create_sensor:
                    # _LOGGER.debug("Device %s: Not tracked (create_sensor=False)", device.name)
                    pass
                elif not self.options.get(CONF_ENABLE_TRILATERATION, True):
                    _LOGGER.debug("Device %s: Trilateration disabled in options", device.name)

            self._refresh_areas_by_min_distance(nowstamp)

            # We might need to freshen deliberately on first start if no new scanners
            # were discovered in the first scan update. This is likely if nothing has changed
            # since the last time we booted.
            # if self._do_full_scanner_init:
            #     if not self._refresh_scanners():
            #         # _LOGGER.debug("Failed to refresh scanners, likely config entry not ready.")
            #         # don't fail the update, just try again next time.
            #         # self.last_update_success = False
            #         pass

            # If any *configured* devices have not yet been seen, create device
            # entries for them so they will claim the restored sensors in HA
            # (this prevents them from restoring at startup as "Unavailable" if they
            # are not currently visible, and will instead show as "Unknown" for
            # sensors and "Away" for device_trackers).
            #
            # This isn't working right if it runs once. Bodge it for now (cost is low)
            # and sort it out when moving to device-based restoration (ie using DR/ER
            # to decide what devices to track and deprecating CONF_DEVICES)
            #
            # if not self._seed_configured_devices_done:
            for _source_address in self.options.get(CONF_DEVICES, []):
                self._get_or_create_device(_source_address)
            self._seed_configured_devices_done = True

            # Trigger creation of any new entities
            #
            # The devices are all updated now (and any new scanners and beacons seen have been added),
            # so let's ensure any devices that we create sensors for are set up ready to go.
            for address, device in self.devices.items():
                if device.create_sensor:
                    if not device.create_all_done:
                        _LOGGER.debug("Firing device_new for %s (%s)", device.name, address)
                        # Note that the below should be OK thread-wise, debugger indicates this is being
                        # called by _run in events.py, so pretty sure we are "in the event loop".
                        async_dispatcher_send(self.hass, SIGNAL_DEVICE_NEW, address)

            # Device Pruning (only runs periodically)
            self.prune_devices()

        finally:
            # end of async update
            self.update_in_progress = False

        self.stamp_last_update_started = nowstamp
        self.stamp_last_update = monotonic_time_coarse()
        self.last_update_success = True
        return result_gather_adverts

    def _async_gather_advert_data(self):
        """Perform the gathering of backend Bluetooth Data and updating scanners and devices."""
        # Initialise ha_scanners if we haven't already
        if self._scanner_init_pending:
            _LOGGER.debug("Scanner init pending, calling _refresh_scanners")
            self._refresh_scanners(force=True)

        for ha_scanner in self._ha_scanners:
            # Create / Get the BermudaDevice for this scanner
            scanner_device = self._get_device(ha_scanner.source)

            if scanner_device is None:
                # Looks like a scanner we haven't met, refresh the list.
                self._refresh_scanners(force=True)
                scanner_device = self._get_device(ha_scanner.source)

            if scanner_device is None:
                # Highly unusual. If we can't find an entry for the scanner
                # maybe it's from an integration that's not yet loaded, or
                # perhaps it's an unexpected type that we don't know how to
                # find.
                _LOGGER_SPAM_LESS.error(
                    f"missing_scanner_entry_{ha_scanner.source}",
                    "Failed to find config for scanner %s, this is probably a bug.",
                    ha_scanner.source,
                )
                continue

            scanner_device.async_as_scanner_update(ha_scanner)

            # Now go through the scanner's adverts and send them to our device objects.
            for bledevice, advertisementdata in ha_scanner.discovered_devices_and_advertisement_data.values():
                if adstamp := scanner_device.async_as_scanner_get_stamp(bledevice.address):
                    if adstamp < self.stamp_last_update_started - 3:
                        # skip older adverts that should already have been processed
                        continue
                if advertisementdata.rssi == -127:
                    # BlueZ is pushing bogus adverts for paired but absent devices.
                    continue

                device = self._get_or_create_device(bledevice.address)
                device.process_advertisement(scanner_device, advertisementdata)

        # end of for ha_scanner loop
        return True

    def prune_devices(self, force_pruning=False):
        """
        Scan through all collected devices, and remove those that meet Pruning criteria.

        By default no pruning will be done if it has been performed within the last
        PRUNE_TIME_INTERVAL, unless the force_pruning flag is set to True.
        """
        if self.stamp_last_prune > monotonic_time_coarse() - PRUNE_TIME_INTERVAL and not force_pruning:
            # We ran recently enough, bail out.
            return
        # stamp the run.
        nowstamp = self.stamp_last_prune = monotonic_time_coarse()
        stamp_known_irk = nowstamp - PRUNE_TIME_KNOWN_IRK
        stamp_unknown_irk = nowstamp - PRUNE_TIME_UNKNOWN_IRK

        # Prune redaction data
        if self.stamp_redactions_expiry is not None and self.stamp_redactions_expiry < nowstamp:
            _LOGGER.debug("Clearing redaction data (%d items)", len(self.redactions))
            self.redactions.clear()
            self.stamp_redactions_expiry = None

        # Prune any IRK MACs that have expired
        self.irk_manager.async_prune()

        # Prune devices.
        prune_list: list[str] = []  # list of addresses to be pruned
        prunable_stamps: dict[str, float] = {}  # dict of potential prunees if we need to be more aggressive.

        metadevice_source_keepers = set()
        for metadevice in self.metadevices.values():
            if len(metadevice.metadevice_sources) > 0:
                # Always keep the most recent source, which we keep in index 0.
                # This covers static iBeacon sources, and possibly IRKs that might exceed
                # the spec lifetime but are going stale because they're away for a bit.
                _first = True
                for address in metadevice.metadevice_sources:
                    if _device := self._get_device(address):
                        if _first or _device.last_seen > stamp_known_irk:
                            # The source has been seen within the spec's limits, keep it.
                            metadevice_source_keepers.add(address)
                            _first = False
                        else:
                            # It's too old to be an IRK, and otherwise we'll auto-detect it,
                            # so let's be rid of it.
                            prune_list.append(address)

        for device_address, device in self.devices.items():
            # Prune any devices that haven't been heard from for too long, but only
            # if we aren't actively tracking them and it's a traditional MAC address.
            # We just collect the addresses first, and do the pruning after exiting this iterator
            #
            # Reduced selection criteria - basically if if's not:
            # - a scanner (beacuse we need those!)
            # - any metadevice less than 15 minutes old (PRUNE_TIME_KNOWN_IRK)
            # - a private_ble device (because they will re-create anyway, plus we auto-sensor them
            # - create_sensor
            # then it should be up for pruning. A stale iBeacon that we don't actually track
            # should totally be pruned if it's no longer around.
            if (
                device_address not in metadevice_source_keepers
                and device not in self.metadevices
                and device_address not in self.scanner_list
                and (not device.create_sensor)  # Not if we track the device
                and (not device.is_scanner)  # redundant, but whatevs.
                and device.address_type != BDADDR_TYPE_NOT_MAC48
            ):
                if device.address_type == BDADDR_TYPE_RANDOM_RESOLVABLE:
                    # This is an *UNKNOWN* IRK source address, or a known one which is
                    # well and truly stale (ie, not in keepers).
                    # We prune unknown irk's aggressively because they pile up quickly
                    # in high-density situations, and *we* don't need to hang on to new
                    # enrollments because we'll seed them from PBLE.
                    if device.last_seen < stamp_unknown_irk:
                        _LOGGER.debug(
                            "Marking stale (%ds) Unknown IRK address for pruning: [%s] %s",
                            nowstamp - device.last_seen,
                            device_address,
                            device.name,
                        )
                        prune_list.append(device_address)
                    elif device.last_seen < nowstamp - 200:  # BlueZ cache time
                        # It's not stale, but we will prune it if we can't make our
                        # quota of PRUNE_MAX_COUNT we'll shave these off too.

                        # Note that because BlueZ doesn't give us timestamps, we guess them
                        # based on whether the rssi has changed. If we delete our existing
                        # device we have nothing to compare too and will forever churn them.
                        # This can change if we drop support for BlueZ or we find a way to
                        # make stamps (we could also just keep a separate list but meh)
                        prunable_stamps[device_address] = device.last_seen

                elif device.last_seen < nowstamp - PRUNE_TIME_DEFAULT:
                    # It's a static address, and stale.
                    _LOGGER.debug(
                        "Marking old device entry for pruning: %s",
                        device.name,
                    )
                    prune_list.append(device_address)
                else:
                    # Device is static, not tracked, not so old, but we might have to prune it anyway
                    prunable_stamps[device_address] = device.last_seen

            # Do nothing else at this level without excluding the keepers first.

        prune_quota_shortfall = len(self.devices) - len(prune_list) - PRUNE_MAX_COUNT
        if prune_quota_shortfall > 0:
            # We need to find more addresses to prune. Perhaps we live
            # in a busy train station, or are under some sort of BLE-MAC
            # DOS-attack.
            if len(prunable_stamps) > 0:
                # Sort the prunables by timestamp ascending
                sorted_addresses = sorted([(v, k) for k, v in prunable_stamps.items()])
                cutoff_index = min(len(sorted_addresses), prune_quota_shortfall)

                _LOGGER.debug(
                    "Prune quota short by %d. Pruning %d extra devices (down to age %0.2f seconds)",
                    prune_quota_shortfall,
                    cutoff_index,
                    nowstamp - sorted_addresses[prune_quota_shortfall - 1][0],
                )
                # pylint: disable-next=unused-variable
                for _stamp, address in sorted_addresses[: prune_quota_shortfall - 1]:
                    prune_list.append(address)
            else:
                _LOGGER.warning(
                    "Need to prune another %s devices to make quota, but no extra prunables available",
                    prune_quota_shortfall,
                )
        else:
            _LOGGER.debug(
                "Pruning %d available MACs, we are inside quota by %d.", len(prune_list), prune_quota_shortfall * -1
            )

        # ###############################################
        # Prune_list is now ready to action. It contains no keepers, and is already
        # expanded if necessary to meet quota, as much as we can.

        # Prune the source devices
        for device_address in prune_list:
            _LOGGER.debug("Acting on prune list for %s", device_address)
            del self.devices[device_address]

        # Clean out the scanners dicts in metadevices and scanners
        # (scanners will have entries if they are also beacons, although
        # their addresses should never get stale, but one day someone will
        # have a beacon that uses randomised source addresses for some reason.
        #
        # Just brute-force all devices, because it was getting a bit hairy
        # ensuring we hit the right ones, and the cost is fairly low and periodic.
        for device in self.devices.values():
            # if (
            #     device.is_scanner
            #     or METADEVICE_PRIVATE_BLE_DEVICE in device.metadevice_type
            #     or METADEVICE_IBEACON_DEVICE in device.metadevice_type
            # ):
            # clean out the metadevice_sources field
            for address in prune_list:
                if address in device.metadevice_sources:
                    device.metadevice_sources.remove(address)

            # clean out the device/scanner advert pairs
            for advert_tuple in list(device.adverts.keys()):
                if device.adverts[advert_tuple].device_address in prune_list:
                    _LOGGER.debug(
                        "Pruning metadevice advert %s aged %ds",
                        advert_tuple,
                        nowstamp - device.adverts[advert_tuple].stamp,
                    )
                    del device.adverts[advert_tuple]

    def discover_private_ble_metadevices(self):
        """
        Access the Private BLE Device integration to find metadevices to track.

        This function sets up the skeleton metadevice entry for Private BLE (IRK)
        devices, ready for update_metadevices to manage.
        """
        if self._do_private_device_init:
            self._do_private_device_init = False
            _LOGGER.debug("Refreshing Private BLE Device list")

            # Iterate through the Private BLE Device integration's entities,
            # and ensure for each "device" we create a source device.
            # pb here means "private ble device"
            pb_entries = self.hass.config_entries.async_entries(DOMAIN_PRIVATE_BLE_DEVICE, include_disabled=False)
            for pb_entry in pb_entries:
                pb_entities = self.er.entities.get_entries_for_config_entry_id(pb_entry.entry_id)
                # This will be a list of entities for a given private ble device,
                # let's pull out the device_tracker one, since it has the state
                # info we need.
                for pb_entity in pb_entities:
                    if pb_entity.domain == Platform.DEVICE_TRACKER:
                        # We found a *device_tracker* entity for the private_ble device.
                        _LOGGER.debug(
                            "Found a Private BLE Device Tracker! %s",
                            pb_entity.entity_id,
                        )

                        # Grab the device entry (for the name, mostly)
                        if pb_entity.device_id is not None:
                            pb_device = self.dr.async_get(pb_entity.device_id)
                        else:
                            pb_device = None

                        # Grab the current state (so we can access the source address attrib)
                        pb_state = self.hass.states.get(pb_entity.entity_id)

                        if pb_state:  # in case it's not there yet
                            pb_source_address = pb_state.attributes.get("current_address", None)
                        else:
                            # Private BLE Device hasn't yet found a source device
                            pb_source_address = None

                        # Get the IRK of the device, which we will use as the address
                        # for the metadevice.
                        # As of 2024.4.0b4 Private_ble appends _device_tracker to the
                        # unique_id of the entity, while we really want to know
                        # the actual IRK, so handle either case by splitting it:
                        _irk = pb_entity.unique_id.split("_")[0]

                        # Create our Meta-Device and tag it up...
                        metadevice = self._get_or_create_device(_irk)
                        # Since user has already configured the Private BLE Device, we
                        # always create sensors for them.
                        metadevice.create_sensor = True

                        # Set a nice name
                        if pb_device:
                            metadevice.name_by_user = pb_device.name_by_user
                            metadevice.name_devreg = pb_device.name
                            metadevice.make_name()

                        # Ensure we track this PB entity so we get source address updates.
                        if pb_entity.entity_id not in self.pb_state_sources:
                            self.pb_state_sources[pb_entity.entity_id] = None  # FIXME: why none?

                        # Add metadevice to list so it gets included in update_metadevices
                        if metadevice.address not in self.metadevices:
                            self.metadevices[metadevice.address] = metadevice

                        if pb_source_address is not None:
                            # We've got a source MAC address!
                            pb_source_address = mac_norm(pb_source_address)

                            # Set up and tag the source device entry
                            source_device = self._get_or_create_device(pb_source_address)
                            source_device.metadevice_type.add(METADEVICE_TYPE_PRIVATE_BLE_SOURCE)

                            # Add source address. Don't remove anything, as pruning takes care of that.
                            if pb_source_address not in metadevice.metadevice_sources:
                                metadevice.metadevice_sources.insert(0, pb_source_address)

                            # Update state_sources so we can track when it changes
                            self.pb_state_sources[pb_entity.entity_id] = pb_source_address

                        else:
                            _LOGGER.debug(
                                "No address available for PB Device %s",
                                pb_entity.entity_id,
                            )

    def register_ibeacon_source(self, source_device: BermudaDevice):
        """
        Create or update the meta-device for tracking an iBeacon.

        This should be called each time we discover a new address advertising
        an iBeacon. This might happen only once at startup, but will also
        happen each time a new MAC address is used by a given iBeacon,
        or each time an existing MAC sends a *new* iBeacon(!)

        This does not update the beacon's details (distance etc), that is done
        in the update_metadevices function after all data has been gathered.
        """
        if METADEVICE_TYPE_IBEACON_SOURCE not in source_device.metadevice_type:
            _LOGGER.error(
                "Only IBEACON_SOURCE devices can be used to see a beacon metadevice. %s is not",
                source_device.name,
            )
        if source_device.beacon_unique_id is None:
            _LOGGER.error("Source device %s is not a valid iBeacon!", source_device.name)
        else:
            metadevice = self._get_or_create_device(source_device.beacon_unique_id)
            if len(metadevice.metadevice_sources) == 0:
                # #### NEW METADEVICE #####
                # (do one-off init stuff here)
                if metadevice.address not in self.metadevices:
                    self.metadevices[metadevice.address] = metadevice

                # Copy over the beacon attributes
                metadevice.name_bt_serviceinfo = source_device.name_bt_serviceinfo
                metadevice.name_bt_local_name = source_device.name_bt_local_name
                metadevice.beacon_unique_id = source_device.beacon_unique_id
                metadevice.beacon_major = source_device.beacon_major
                metadevice.beacon_minor = source_device.beacon_minor
                metadevice.beacon_power = source_device.beacon_power
                metadevice.beacon_uuid = source_device.beacon_uuid

                # Check if we should set up sensors for this beacon
                if metadevice.address.upper() in self.options.get(CONF_DEVICES, []):
                    # This is a meta-device we track. Flag it for set-up:
                    metadevice.create_sensor = True

            # #### EXISTING METADEVICE ####
            # (only do things that might have to change when MAC address cycles etc)

            if source_device.address not in metadevice.metadevice_sources:
                # We have a *new* source device.
                # insert this device as a known source
                metadevice.metadevice_sources.insert(0, source_device.address)

                # If we have a new / better name, use that..
                metadevice.name_bt_serviceinfo = metadevice.name_bt_serviceinfo or source_device.name_bt_serviceinfo
                metadevice.name_bt_local_name = metadevice.name_bt_local_name or source_device.name_bt_local_name

    def update_metadevices(self):
        """
        Create or update iBeacon, Private_BLE and other meta-devices from
        the received advertisements.

        This must be run on each update cycle, after the calculations for each source
        device is done, since we will copy their results into the metadevice.

        Area matching and trilateration will be performed *after* this, as they need
        to consider the full collection of sources, not just the ones of a single
        source device.
        """
        # First seed the Private BLE metadevice skeletons. It will only do anything
        # if the self._do_private_device_init flag is set.
        # FIXME: Can we delete this? pble's should create at realtime as they
        # are detected now.
        self.discover_private_ble_metadevices()

        # iBeacon devices should already have their metadevices created, so nothing more to
        # set up for them.

        for metadevice in self.metadevices.values():
            # Find every known source device and copy their adverts in.

            # Keep track of whether we want to recalculate the name fields at the end.
            _want_name_update = False
            _sources_to_remove = []

            for source_address in metadevice.metadevice_sources:
                # Get the BermudaDevice holding those adverts
                # TODO: Verify it's OK to not create here. Problem is that if we do create,
                # it causes a binge/purge cycle during pruning since it has no adverts on it.
                source_device = self._get_device(source_address)
                if source_device is None:
                    # No ads current in the backend for this one. Not an issue, the mac might be old
                    # or now showing up yet.
                    # _LOGGER_SPAM_LESS.debug(
                    #     f"metaNoAdsFor_{metadevice.address}_{source_address}",
                    #     "Metadevice %s: no adverts for source MAC %s found during update_metadevices",
                    #     metadevice.__repr__(),
                    #     source_address,
                    # )
                    continue

                if (
                    METADEVICE_IBEACON_DEVICE in metadevice.metadevice_type
                    and metadevice.beacon_unique_id != source_device.beacon_unique_id
                ):
                    # This source device no longer has the same ibeacon uuid+maj+min as
                    # the metadevice has.
                    # Some iBeacons (specifically Bluecharms) change uuid on movement.
                    #
                    # This source device has changed its uuid, so we won't track it against
                    # this metadevice any more / for now, and we will also remove
                    # the existing scanner entries on the metadevice, to ensure it goes
                    # `unknown` immediately (assuming no other source devices show up)
                    #
                    # Note that this won't quick-away devices that change their MAC at the
                    # same time as changing their uuid (like manually altering the beacon
                    # in an Android 15+), since the old source device will still be a match.
                    # and will be subject to the nomal DEVTRACK_TIMEOUT.
                    #
                    _LOGGER.debug(
                        "Source %s for metadev %s changed iBeacon identifiers, severing", source_device, metadevice
                    )
                    for key_address, key_scanner in list(metadevice.adverts):
                        if key_address == source_device.address:
                            del metadevice.adverts[(key_address, key_scanner)]
                    if source_device.address in metadevice.metadevice_sources:
                        # Remove this source from the list once we're done iterating on it
                        _sources_to_remove.append(source_device.address)
                    continue  # to next metadevice_source

                # Copy every ADVERT_TUPLE into our metadevice
                for advert_tuple in source_device.adverts:
                    metadevice.adverts[advert_tuple] = source_device.adverts[advert_tuple]

                # Update last_seen if the source is newer.
                if metadevice.last_seen < source_device.last_seen:
                    metadevice.last_seen = source_device.last_seen

                # If not done already, set the source device's ref_power from our own. This will cause
                # the source device and all its scanner entries to update their
                # distance measurements. This won't affect Area wins though, because
                # they are "relative", not absolute.

                # FIXME: This has two potential bugs:
                # - if multiple metadevices share a source, they will
                #   "fight" over their preferred ref_power, if different.
                # - The non-meta device (if tracked) will receive distances
                #   based on the meta device's ref_power.
                # - The non-meta device if tracked will have its own ref_power ignored.
                #
                # None of these are terribly awful, but worth fixing.

                # Note we are setting the ref_power on the source_device, not the
                # individual scanner entries (it will propagate to them though)
                if source_device.ref_power != metadevice.ref_power:
                    source_device.set_ref_power(metadevice.ref_power)

                # anything that isn't already set to something interesting, overwrite
                # it with the new device's data.
                for key, val in source_device.items():
                    if val in [
                        source_device.name_bt_local_name,
                        source_device.name_bt_serviceinfo,
                        source_device.manufacturer,
                    ] and metadevice[key] in [None, False]:
                        metadevice[key] = val
                        _want_name_update = True

                if _want_name_update:
                    metadevice.make_name()

                # Anything that's VERY interesting, overwrite it regardless of what's already there:
                # INTERESTING:
                for key, val in source_device.items():
                    if val in [
                        source_device.beacon_major,
                        source_device.beacon_minor,
                        source_device.beacon_power,
                        source_device.beacon_unique_id,
                        source_device.beacon_uuid,
                    ]:
                        metadevice[key] = val
                        # _want_name_update = True
            # Done iterating sources, remove any to be dropped
            for source in _sources_to_remove:
                metadevice.metadevice_sources.remove(source)
            if _want_name_update:
                metadevice.make_name()

    def dt_mono_to_datetime(self, stamp) -> datetime:
        """Given a monotonic timestamp, convert to datetime object."""
        age = monotonic_time_coarse() - stamp
        return now() - timedelta(seconds=age)

    def dt_mono_to_age(self, stamp) -> str:
        """Convert monotonic timestamp to age (eg: "6 seconds ago")."""
        return get_age(self.dt_mono_to_datetime(stamp))

    def resolve_area_name(self, area_id) -> str | None:
        """
        Given an area_id, return the current area name.

        Will return None if the area id does *not* resolve to a single
        known area name.
        """
        areas = self.ar.async_get_area(area_id)
        if hasattr(areas, "name"):
            return getattr(areas, "name", "invalid_area")
        return None

    def _refresh_areas_by_min_distance(self, nowstamp: float):
        """Set area for ALL devices based on closest beacon."""
        for device in self.devices.values():
            if (
                # device.is_scanner is not True  # exclude scanners.
                device.create_sensor  # include any devices we are tracking
                # or device.metadevice_type in METADEVICE_SOURCETYPES  # and any source devices for PBLE, ibeacon etc
            ):
                # Skip devices with fresh trilateration positions when override is enabled
                if (
                    self.options.get(CONF_TRILATERATION_OVERRIDE_AREA, True)
                    and device.position_timestamp is not None
                    and device.calculated_position is not None
                    and device.area_id is not None
                    and nowstamp - device.position_timestamp <= TRILATERATION_POSITION_TIMEOUT
                ):
                    _LOGGER.debug(
                        "Device %s: Skipping min-distance area assignment - trilateration owns area (position age: %.1fs)",
                        device.name,
                        nowstamp - device.position_timestamp,
                    )
                    continue

                # Use min-distance for devices without trilateration or with stale positions
                if device.position_timestamp is not None and nowstamp - device.position_timestamp > TRILATERATION_POSITION_TIMEOUT:
                    _LOGGER.debug(
                        "Device %s: Using min-distance - trilateration position stale (%.1fs old)",
                        device.name,
                        nowstamp - device.position_timestamp,
                    )

                self._refresh_area_by_min_distance(device)

    @dataclass
    class AreaTests:
        """
        Holds the results of Area-based tests.

        Likely to become a stand-alone class for performing the whole area-selection
        process.
        """

        device: str = ""
        scannername: tuple[str, str] = ("", "")
        areas: tuple[str, str] = ("", "")
        pcnt_diff: float = 0  # distance percentage difference.
        same_area: bool = False  # The old scanner is in the same area as us.
        # last_detection: tuple[float, float] = (0, 0)  # bt manager's last_detection field. Compare with ours.
        last_ad_age: tuple[float, float] = (0, 0)  # seconds since we last got *any* ad from scanner
        this_ad_age: tuple[float, float] = (0, 0)  # how old the *current* advert is on this scanner
        distance: tuple[float, float] = (0, 0)
        hist_min_max: tuple[float, float] = (0, 0)  # min/max distance from history
        # velocity: tuple[float, float] = (0, 0)
        # last_closer: tuple[float, float] = (0, 0)  # since old was closer and how long new has been closer
        reason: str | None = None  # reason/result

        def sensortext(self) -> str:
            """Return a text summary suitable for use in a sensor entity."""
            out = ""
            for var, val in vars(self).items():
                out += f"{var}|"
                if isinstance(val, tuple):
                    for v in val:
                        if isinstance(v, float):
                            out += f"{v:.2f}|"
                        else:
                            out += f"{v}"
                    # out += "\n"
                elif var == "pcnt_diff":
                    out += f"{val:.3f}"
                else:
                    out += f"{val}"
                out += "\n"
            return out[:255]

        def __str__(self) -> str:
            """
            Create string representation for easy debug logging/dumping
            and potentially a sensor for logging Area decisions.
            """
            out = ""
            for var, val in vars(self).items():
                out += f"** {var:20} "
                if isinstance(val, tuple):
                    for v in val:
                        if isinstance(v, float):
                            out += f"{v:.2f} "
                        else:
                            out += f"{v} "
                    out += "\n"
                elif var == "pcnt_diff":
                    out += f"{val:.3f}\n"
                else:
                    out += f"{val}\n"
            return out

    def _refresh_area_by_min_distance(self, device: BermudaDevice):
        """Very basic Area setting by finding closest proxy to a given device."""
        # The current area_scanner (which might be None) is the one to beat.
        incumbent: BermudaAdvert | None = device.area_advert

        _max_radius = self.options.get(CONF_MAX_RADIUS, DEFAULT_MAX_RADIUS)
        nowstamp = monotonic_time_coarse()

        tests = self.AreaTests()
        tests.device = device.name

        _superchatty = False  # Set to true for very verbose logging about area wins
        # if device.name in ("Ash Pixel IRK", "Garage", "Melinda iPhone"):
        #     _superchatty = True

        # Sort scanners deterministically: freshest first, then strongest signal (lowest RSSI distance)
        # This ensures consistent results regardless of dict iteration order
        sorted_adverts = sorted(
            device.adverts.values(),
            key=lambda adv: (
                -(adv.stamp if adv.stamp is not None else 0),  # Negative for descending (freshest first)
                adv.rssi_distance if adv.rssi_distance is not None else float("inf"),  # Ascending (closest first)
            ),
        )

        for challenger in sorted_adverts:
            # Check each scanner and any time one is found to be closer / better than
            # the existing closest_scanner, replace it. At the end we should have the
            # right one. In theory.
            #
            # Note that rssi_distance is smoothed/filtered, and might be None if the last
            # reading was old enough that our algo decides it's "away".
            #
            # Every loop, every test is just a two-way race.

            # Is the challenger an invalid contender?
            if (
                # no competing against ourselves...
                incumbent is challenger  # no competing against ourselves.
            ):
                continue

            # No winning with stale adverts. If we didn't win back when it was fresh,
            # we've no business winning now. This guards against a single advert
            # being reported by two proxies at slightly different times, and the area
            # switching to the later one after the reading times out on the first.
            # The timeout value is fairly arbitrary, if it's too small then we risk
            # ignoring valid reports from slow proxies (or if our processing loop is
            # delayed / lengthened). Too long and we add needless jumping around for a
            # device that isn't actually being actively detected.
            if challenger.stamp < nowstamp - AREA_MAX_AD_AGE:
                # our ad is too old.
                continue

            # If we are too far away or don't have an area, we cannot win...
            if (
                challenger.rssi_distance is None
                or challenger.rssi_distance > _max_radius
                or challenger.area_id is None
            ):
                continue

            # At this point the challenger is a vaild contender...

            # Is the incumbent a valid contender?

            # If closest scanner lacks critical data, we win.
            if (
                incumbent is None
                or incumbent.rssi_distance is None
                or incumbent.area_id is None
                # Extra checks that are redundant but make linting easier later...
                # or closest_advert.hist_distance_by_interval is None
            ):
                # Default Instawin!
                incumbent = challenger
                if _superchatty:
                    _LOGGER.debug(
                        "%s IS closesr to %s: Encumbant is invalid",
                        device.name,
                        challenger.name,
                    )
                continue

            # NOTE:
            # From here on in, don't award a win directly. Instead award a loss if the new scanner is
            # not a contender, but otherwise build a set of test scores and make a determination at the
            # end.

            # If we ARE NOT ACTUALLY CLOSER(!) we can not win.
            if incumbent.rssi_distance < challenger.rssi_distance:
                # we are not even closer!
                continue

            tests.reason = None  # ensure we don't trigger logging if no decision was made.
            tests.same_area = incumbent.area_id == challenger.area_id
            tests.areas = (incumbent.area_name or "", challenger.area_name or "")
            tests.scannername = (incumbent.name, challenger.name)
            tests.distance = (incumbent.rssi_distance, challenger.rssi_distance)
            # tests.velocity = (
            #     next((val for val in closest_scanner.hist_velocity), 0),
            #     next((val for val in scanner.hist_velocity), 0),
            # )

            # How recently have we heard from the scanners themselves (not just for this device's adverts)?
            tests.last_ad_age = (
                nowstamp - incumbent.scanner_device.last_seen,
                nowstamp - challenger.scanner_device.last_seen,
            )

            # How old are the ads?
            tests.this_ad_age = (
                nowstamp - incumbent.stamp,
                nowstamp - challenger.stamp,
            )

            # Calculate the percentage difference between the challenger and incumbent's distances
            _pda = challenger.rssi_distance
            _pdb = incumbent.rssi_distance
            tests.pcnt_diff = abs(_pda - _pdb) / ((_pda + _pdb) / 2)

            # Same area. Confirm freshness and distance.
            if (
                tests.same_area
                and (tests.this_ad_age[0] > tests.this_ad_age[1] + 1)
                and tests.distance[0] >= tests.distance[1]
            ):
                tests.reason = "WIN awarded for same area, newer, closer advert"
                incumbent = challenger
                continue

            # Hysteresis.
            # If our worst reading in max_seconds is still closer than the incumbent's **best** reading
            # in that time, and we are over a PD threshold, we win.
            #
            min_history = 3  # we must have at least this much history
            history_window = 5  # the time period to compare between us and incumbent
            pdiff_outright = 0.30  # Percentage difference to win outright / instantly
            pdiff_historical = 0.15  # Percentage difference required to win on historical test
            if len(challenger.hist_distance_by_interval) > min_history:  # we have enough history, let's go..
                # Use islice for deque slicing (deques don't support [:N] notation)
                from itertools import islice
                tests.hist_min_max = (
                    min(islice(incumbent.hist_distance_by_interval, 0, history_window)),  # The closest that the incumbent has been
                    max(islice(challenger.hist_distance_by_interval, 0, history_window)),  # The **furthest** we have been in that time
                )
                if (
                    tests.hist_min_max[1] < tests.hist_min_max[0]
                    and tests.pcnt_diff > pdiff_historical  # and we're significantly closer.
                ):
                    tests.reason = "WIN on historical min/max"
                    incumbent = challenger
                    continue

            if tests.pcnt_diff < pdiff_outright:
                # Didn't make the cut. We're not "different enough" given how
                # recently the previous nearest was updated.
                tests.reason = "LOSS - failed on percentage_difference"
                continue

            # If we made it through all of that, we're winning, so far!
            tests.reason = "WIN by not losing!"

            incumbent = challenger

        if _superchatty and tests.reason is not None:
            _LOGGER.info(
                "***************\n**************** %s *******************\n%s",
                tests.reason,
                tests,
            )

        # Debug log showing all scanner candidates and selection decision
        if _LOGGER.isEnabledFor(logging.DEBUG):
            scanner_summary = []
            for adv in sorted_adverts[:5]:  # Show top 5 candidates
                scanner = self.devices.get(adv.scanner_address)
                if scanner and adv.rssi_distance is not None:
                    age = nowstamp - adv.stamp if adv.stamp else 999
                    status = "STALE" if age > AREA_MAX_AD_AGE else "fresh"
                    status = "SELECTED" if adv is incumbent else status
                    scanner_summary.append(
                        f"{scanner.name}: {adv.rssi_distance:.1f}m @ {adv.rssi}dBm ({status}, {age:.1f}s old)"
                    )
            if scanner_summary:
                _LOGGER.debug(
                    "Device %s scanner candidates: %s",
                    device.name,
                    " | ".join(scanner_summary),
                )

        _superchatty = False

        if device.area_advert != incumbent and tests.reason is not None:
            device.diag_area_switch = tests.sensortext()

        # Apply the newly-found closest scanner (or apply None if we didn't find one)
        device.apply_scanner_selection(incumbent)

    def _refresh_scanners(self, force=False):
        """
        Refresh data on existing scanner objects, and rebuild if scannerlist has changed.

        Called on every update cycle, this handles the *fast* updates (such as updating
        timestamps). If it detects that the list of scanners has changed (or is called
        with force=True) then the full list of scanners will be rebuild by calling
        _rebuild_scanners.
        """
        self._rebuild_scanner_list(force=force)

    def _rebuild_scanner_list(self, force=False):
        """
        Rebuild Bermuda's internal list of scanners.

        Called on every update (via _refresh_scanners) but exits *quickly*
        *unless*:
          - the scanner set has changed or
          - force=True or
          - self._force_full_scanner_init=True
        """
        # _new_ha_scanners = set[BaseHaScanner]
        # Using new API in 2025.2
        _new_ha_scanners: set[BaseHaScanner] = set(self._manager.async_current_scanners())

        if _new_ha_scanners is self._ha_scanners or _new_ha_scanners == self._ha_scanners:
            # No changes, but if we're in init phase, mark as complete
            if self._scanner_init_pending:
                _LOGGER.info("Scanner initialization complete (no changes detected), enabling position loading")
                self._scanner_init_pending = False
            return

        _LOGGER.debug("HA Base Scanner Set has changed, rebuilding.")
        self._ha_scanners = _new_ha_scanners

        self._async_purge_removed_scanners()

        # So we can raise a single repair listing all area-less scanners:
        _scanners_without_areas: list[str] = []

        # Find active HaBaseScanners in the backend and treat that as our
        # authoritative source of truth.
        #
        for hascanner in self._ha_scanners:
            scanner_address = mac_norm(hascanner.source)
            bermuda_scanner = self._get_or_create_device(scanner_address)
            bermuda_scanner.async_as_scanner_init(hascanner)

            if bermuda_scanner.area_id is None:
                _scanners_without_areas.append(f"{bermuda_scanner.name} [{bermuda_scanner.address}]")
        # TODO: map to area from the map, to not give false positives
        self._async_manage_repair_scanners_without_areas(_scanners_without_areas)

        # Mark scanner initialization as complete
        if self._scanner_init_pending:
            _LOGGER.info("Scanner initialization complete, enabling position loading")
            self._scanner_init_pending = False
            # Load scanner positions on first initialization
            nowstamp = monotonic_time_coarse()
            _LOGGER.info("Loading scanner positions after initial scanner discovery")
            self.load_scanner_positions()
            self._scanner_positions_loaded = True
            self.stamp_last_position_load = nowstamp

    def _async_purge_removed_scanners(self):
        """Demotes any devices that are no longer scanners based on new self.hascanners."""
        _scanners = [device.address for device in self.devices.values() if device.is_scanner]
        for ha_scanner in self._ha_scanners:
            scanner_address = mac_norm(ha_scanner.source)
            if scanner_address in _scanners:
                # This is still an extant HA Scanner, so we'll keep it.
                _scanners.remove(scanner_address)
        # Whatever's left are presumably no longer scanners.
        for address in _scanners:
            _LOGGER.info("Demoting ex-scanner %s", self.devices[address].name)
            self.devices[address].async_as_scanner_nolonger()

    def _async_manage_repair_scanners_without_areas(self, scannerlist: list[str]):
        """
        Raise a repair for any scanners that lack an area assignment.

        This function will take care of ensuring a repair is (re)raised
        or cleared (if the list is empty) when given a list of area-less scanner names.

        scannerlist should contain a friendly string to name each scanner missing an area.
        """
        if self._scanners_without_areas != scannerlist:
            self._scanners_without_areas = scannerlist
            # Clear any existing repair, because it's either resolved now (empty list) or we need to re-issue
            # the repair in order to update the scanner list (re-calling doesn't update it).
            ir.async_delete_issue(self.hass, DOMAIN, REPAIR_SCANNER_WITHOUT_AREA)

            if self._scanners_without_areas and len(self._scanners_without_areas) != 0:
                ir.async_create_issue(
                    self.hass,
                    DOMAIN,
                    REPAIR_SCANNER_WITHOUT_AREA,
                    translation_key=REPAIR_SCANNER_WITHOUT_AREA,
                    translation_placeholders={
                        "scannerlist": "".join(f"- {name}\n" for name in self._scanners_without_areas),
                    },
                    severity=ir.IssueSeverity.ERROR,
                    is_fixable=False,
                )

    # *** Not required now that we don't reload for scanners.
    # @callback
    # def async_call_update_entry(self, confdata_scanners) -> None:
    #     """
    #     Call in the event loop to update the scanner entries in our config.

    #     We do this via add_job to ensure it runs in the event loop.
    #     """
    #     # Clear the flag for init and update the stamp
    #     self._do_full_scanner_init = False
    #     self.last_config_entry_update = monotonic_time_coarse()
    #     # Apply new config (will cause reload if there are changes)
    #     self.hass.config_entries.async_update_entry(
    #         self.config_entry,
    #         data={
    #             **self.config_entry.data,
    #             CONFDATA_SCANNERS: confdata_scanners,
    #         },
    #     )

    async def service_dump_devices(self, call: ServiceCall) -> ServiceResponse:  # pylint: disable=unused-argument;
        """Return a dump of beacon advertisements by receiver."""
        out = {}
        addresses_input = call.data.get("addresses", "")
        redact = call.data.get("redact", False)
        configured_devices = call.data.get("configured_devices", False)

        # Choose filter for device/address selection
        addresses = []
        if addresses_input != "":
            # Specific devices
            addresses += addresses_input.upper().split()
        if configured_devices:
            # configured and scanners
            addresses += self.scanner_list
            addresses += self.options.get(CONF_DEVICES, [])
            # known IRK/Private BLE Devices
            addresses += self.pb_state_sources

        # lowercase all the addresses for matching
        addresses = list(map(str.lower, addresses))

        # Build the dict of devices
        for address, device in self.devices.items():
            if len(addresses) == 0 or address.lower() in addresses:
                out[address] = device.to_dict()

        if redact:
            _stamp_redact = monotonic_time_coarse()
            out = cast("ServiceResponse", self.redact_data(out))
            _stamp_redact_elapsed = monotonic_time_coarse() - _stamp_redact
            if _stamp_redact_elapsed > 3:  # It should be fast now.
                _LOGGER.warning("Dump devices redaction took %2f seconds", _stamp_redact_elapsed)
            else:
                _LOGGER.debug("Dump devices redaction took %2f seconds", _stamp_redact_elapsed)
        return out

    async def service_check_scanner_broadcasting(self, call: ServiceCall) -> ServiceResponse:  # pylint: disable=unused-argument;
        """
        Check which scanners are broadcasting BLE advertisements.

        Returns detailed information about scanner broadcasting status for
        scanner-to-scanner calibration diagnostics.
        """
        scanner_devices = {addr: dev for addr, dev in self.devices.items() if dev.is_scanner}
        
        result = {
            "total_scanners": len(scanner_devices),
            "broadcasting": [],
            "not_broadcasting": [],
            "scanner_pairs": [],
            "summary": "",
        }

        if len(scanner_devices) < 2:
            result["summary"] = f"Only {len(scanner_devices)} scanner(s) found. Need at least 2 for scanner-to-scanner calibration."
            return result

        broadcasting_info = {}
        non_broadcasting = []

        # Check each scanner for broadcasting
        for scanner_addr, scanner_device in scanner_devices.items():
            receivers = []
            
            # Check if any other device has received from this scanner
            for device_addr, device in self.devices.items():
                if device_addr == scanner_addr:
                    continue
                
                for advert_key in device.adverts.keys():
                    if isinstance(advert_key, tuple) and len(advert_key) == 2:
                        transmitting_device, receiving_scanner = advert_key
                        
                        if transmitting_device == scanner_addr and receiving_scanner in scanner_devices:
                            receivers.append(receiving_scanner)
            
            if receivers:
                broadcasting_info[scanner_addr] = receivers
                result["broadcasting"].append({
                    "address": scanner_addr,
                    "name": scanner_device.name,
                    "seen_by": len(receivers),
                    "receivers": receivers,
                })
            else:
                non_broadcasting.append(scanner_device)
                result["not_broadcasting"].append({
                    "address": scanner_addr,
                    "name": scanner_device.name,
                    "type": "ESPHome" if "espresence" in scanner_device.name.lower() or "esp" in scanner_device.name.lower() else "Unknown",
                })

        # Collect scanner pair information
        min_samples = self.options.get(CONF_AUTO_CALIBRATION_SAMPLES, 50)
        scanner_pairs_rssi = self.collect_scanner_to_scanner_rssi_data(min_samples=min_samples)
        
        for (receiver, transmitter), rssi_samples in scanner_pairs_rssi.items():
            result["scanner_pairs"].append({
                "receiver": receiver,
                "transmitter": transmitter,
                "samples": len(rssi_samples),
                "avg_rssi": sum(rssi_samples) / len(rssi_samples) if rssi_samples else 0,
            })

        # Generate summary
        broadcasting_count = len(broadcasting_info)
        non_broadcasting_count = len(non_broadcasting)
        pairs_count = len(scanner_pairs_rssi)
        min_pairs = AUTO_CALIBRATION_MIN_SCANNER_PAIRS

        if pairs_count >= min_pairs:
            result["summary"] = (
                f"✓ Ready for scanner-to-scanner calibration! "
                f"{broadcasting_count}/{len(scanner_devices)} scanners broadcasting, "
                f"{pairs_count} scanner pairs found (need {min_pairs})."
            )
        elif broadcasting_count == 0:
            result["summary"] = (
                f"⚠ No scanners are broadcasting. All {len(scanner_devices)} scanners need ESPHome configuration to broadcast BLE advertisements. "
                f"See 'not_broadcasting' list for details."
            )
        elif non_broadcasting_count > 0:
            result["summary"] = (
                f"⚠ Only {broadcasting_count}/{len(scanner_devices)} scanners broadcasting, "
                f"found {pairs_count} pairs (need {min_pairs}). "
                f"{non_broadcasting_count} scanner(s) need ESPHome configuration: " +
                ", ".join([dev.name for dev in non_broadcasting[:3]]) +
                (f" (+{non_broadcasting_count - 3} more)" if non_broadcasting_count > 3 else "")
            )
        else:
            result["summary"] = (
                f"⚠ {broadcasting_count} scanners broadcasting but only {pairs_count} pairs found (need {min_pairs}). "
                f"Wait for more RSSI samples to accumulate (need {min_samples} per pair)."
            )

        return result

    def redaction_list_update(self):
        """
        Freshen or create the list of match/replace pairs that we use to
        redact MAC addresses. This gives a set of helpful address replacements
        that still allows identifying device entries without disclosing MAC
        addresses.
        """
        _stamp = monotonic_time_coarse()

        # counter for incrementing replacement names (eg, SCANNER_n). The length
        # of the existing redaction list is a decent enough starting point.
        i = len(self.redactions)

        # SCANNERS
        for non_lower_address in self.scanner_list:
            address = non_lower_address.lower()
            if address not in self.redactions:
                i += 1
                for altmac in mac_explode_formats(address):
                    self.redactions[altmac] = f"{address[:2]}::SCANNER_{i}::{address[-2:]}"
        _LOGGER.debug("Redact scanners: %ss, %d items", monotonic_time_coarse() - _stamp, len(self.redactions))
        # CONFIGURED DEVICES
        for non_lower_address in self.options.get(CONF_DEVICES, []):
            address = non_lower_address.lower()
            if address not in self.redactions:
                i += 1
                if address.count("_") == 2:
                    self.redactions[address] = f"{address[:4]}::CFG_iBea_{i}::{address[32:]}"
                    # Raw uuid in advert
                    self.redactions[address.split("_")[0]] = f"{address[:4]}::CFG_iBea_{i}_{address[32:]}::"
                elif len(address) == 17:
                    for altmac in mac_explode_formats(address):
                        self.redactions[altmac] = f"{address[:2]}::CFG_MAC_{i}::{address[-2:]}"
                else:
                    # Don't know what it is, but not a mac.
                    self.redactions[address] = f"CFG_OTHER_{1}_{address}"
        _LOGGER.debug("Redact confdevs: %ss, %d items", monotonic_time_coarse() - _stamp, len(self.redactions))
        # EVERYTHING ELSE
        for non_lower_address, device in self.devices.items():
            address = non_lower_address.lower()
            if address not in self.redactions:
                # Only add if they are not already there.
                i += 1
                if device.address_type == ADDR_TYPE_PRIVATE_BLE_DEVICE:
                    self.redactions[address] = f"{address[:4]}::IRK_DEV_{i}"
                elif address.count("_") == 2:
                    self.redactions[address] = f"{address[:4]}::OTHER_iBea_{i}::{address[32:]}"
                    # Raw uuid in advert
                    self.redactions[address.split("_")[0]] = f"{address[:4]}::OTHER_iBea_{i}_{address[32:]}::"
                elif len(address) == 17:  # a MAC
                    for altmac in mac_explode_formats(address):
                        self.redactions[altmac] = f"{address[:2]}::OTHER_MAC_{i}::{address[-2:]}"
                else:
                    # Don't know what it is.
                    self.redactions[address] = f"OTHER_{i}_{address}"
        _LOGGER.debug("Redact therest: %ss, %d items", monotonic_time_coarse() - _stamp, len(self.redactions))
        _elapsed = monotonic_time_coarse() - _stamp
        if _elapsed > 0.5:
            _LOGGER.warning("Redaction list update took %.3f seconds, has %d items", _elapsed, len(self.redactions))
        else:
            _LOGGER.debug("Redaction list update took %.3f seconds, has %d items", _elapsed, len(self.redactions))
        self.stamp_redactions_expiry = monotonic_time_coarse() + PRUNE_TIME_REDACTIONS

    def redact_data(self, data, first_recursion=True):
        """
        Wash any collection of data of any MAC addresses.

        Uses the redaction list of substitutions if already created, then
        washes any remaining mac-like addresses. This routine is recursive,
        so if you're changing it bear that in mind!
        """
        if first_recursion:
            # On first/outer call, refresh the redaction list to ensure
            # we don't let any new addresses slip through. Might be expensive
            # on first call, but will be much cheaper for subsequent calls.
            self.redaction_list_update()
            first_recursion = False

        if isinstance(data, str):  # Base Case
            datalower = data.lower()
            # the end of the recursive wormhole, do the actual work:
            if datalower in self.redactions:
                # Full string match, a quick short-circuit
                data = self.redactions[datalower]
            else:
                # Search for any of the redaction strings in the data.
                for find, fix in list(self.redactions.items()):
                    if find in datalower:
                        data = datalower.replace(find, fix)
                        # don't break out because there might be multiple fixes required.
            # redactions done, now replace any remaining MAC addresses
            # We are only looking for xx:xx:xx... format.
            return self._redact_generic_re.sub(self._redact_generic_sub, data)
        elif isinstance(data, dict):
            return {self.redact_data(k, False): self.redact_data(v, False) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.redact_data(v, False) for v in data]
        else:  # Base Case
            return data
