"""Adds config flow for Bermuda BLE Trilateration."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import voluptuous as vol
from bluetooth_data_tools import monotonic_time_coarse
from homeassistant import config_entries
from homeassistant.config_entries import OptionsFlowWithConfigEntry
from homeassistant.core import callback
from homeassistant.helpers import device_registry as dr
from homeassistant.helpers.selector import DeviceSelector
from homeassistant.helpers.selector import DeviceSelectorConfig
from homeassistant.helpers.selector import ObjectSelector
from homeassistant.helpers.selector import SelectOptionDict
from homeassistant.helpers.selector import SelectSelector
from homeassistant.helpers.selector import SelectSelectorConfig
from homeassistant.helpers.selector import SelectSelectorMode
from homeassistant.helpers.selector import TextSelector
from homeassistant.helpers.selector import TextSelectorConfig
from homeassistant.helpers.selector import TextSelectorType

from .const import ADDR_TYPE_IBEACON
from .const import ADDR_TYPE_PRIVATE_BLE_DEVICE
from .const import BDADDR_TYPE_RANDOM_RESOLVABLE
from .const import CONF_ATTENUATION
from .const import CONF_DEVICES
from .const import CONF_DEVTRACK_TIMEOUT
from .const import CONF_IMPORT_MODE
from .const import CONF_JSON_IMPORT
from .const import CONF_MAX_RADIUS
from .const import CONF_MAX_VELOCITY
from .const import CONF_REF_POWER
from .const import CONF_RSSI_OFFSETS
from .const import CONF_SAVE_AND_CLOSE
from .const import CONF_SCANNER_INFO
from .const import CONF_SCANNERS
from .const import CONF_SMOOTHING_SAMPLES
from .const import CONF_UPDATE_INTERVAL
from .const import CONFDATA_FLOORS
from .const import CONFDATA_ROOMS
from .const import CONFDATA_SCANNER_POSITIONS
from .const import DEFAULT_ATTENUATION
from .const import DEFAULT_DEVTRACK_TIMEOUT
from .const import DEFAULT_MAX_RADIUS
from .const import DEFAULT_MAX_VELOCITY
from .const import DEFAULT_REF_POWER
from .const import DEFAULT_SMOOTHING_SAMPLES
from .const import DEFAULT_UPDATE_INTERVAL
from .const import DISTANCE_INFINITE
from .const import DOMAIN
from .const import DOMAIN_PRIVATE_BLE_DEVICE
from .const import NAME
from .util import mac_redact
from .util import rssi_to_metres

if TYPE_CHECKING:
    from homeassistant.components.bluetooth import BluetoothServiceInfoBleak
    from homeassistant.config_entries import ConfigFlowResult

    from . import BermudaConfigEntry
    from .bermuda_device import BermudaDevice
    from .coordinator import BermudaDataUpdateCoordinator

# from homeassistant import data_entry_flow

# from homeassistant.helpers.aiohttp_client import async_create_clientsession


class BermudaFlowHandler(config_entries.ConfigFlow, domain=DOMAIN):
    """Config flow for bermuda."""

    VERSION = 1
    # CONNECTION_CLASS = config_entries.CONN_CLASS_CLOUD_POLL

    def __init__(self) -> None:
        """Initialize."""
        self._errors = {}

    async def async_step_bluetooth(self, discovery_info: BluetoothServiceInfoBleak) -> ConfigFlowResult:
        """
        Support automatic initiation of setup through bluetooth discovery.
        (we still show a confirmation form to the user, though)
        This is triggered by discovery matchers set in manifest.json,
        and since we track any BLE advert, we're being a little cheeky by listing any.
        """
        if self._async_current_entries():
            return self.async_abort(reason="single_instance_allowed")

        # Create a unique ID so that we don't get multiple discoveries appearing.
        await self.async_set_unique_id(DOMAIN)
        self._abort_if_unique_id_configured()

        return self.async_show_form(step_id="user", description_placeholders={"name": NAME})

    async def async_step_user(self, user_input=None):
        """
        Handle a flow initialized by the user.

        We don't need any config for base setup, so we just activate
        (but only for one instance)
        """
        if self._async_current_entries():
            return self.async_abort(reason="single_instance_allowed")

        if user_input is not None:
            # create the integration!
            return self.async_create_entry(title=NAME, data={"source": "user"}, description=NAME)

        return self.async_show_form(step_id="user", description_placeholders={"name": NAME})

    @staticmethod
    @callback
    def async_get_options_flow(config_entry):
        return BermudaOptionsFlowHandler(config_entry)

    # async def _show_config_form(self, user_input):  # pylint: disable=unused-argument
    #     """Show the configuration form to edit location data."""
    #     return self.async_show_form(
    #         step_id="user",
    #         data_schema=vol.Schema(
    #             {vol.Required(CONF_USERNAME): str, vol.Required(CONF_PASSWORD): str}
    #         ),
    #         errors=self._errors,
    #     )


class BermudaOptionsFlowHandler(OptionsFlowWithConfigEntry):
    """Config flow options handler for bermuda."""

    def __init__(self, config_entry: BermudaConfigEntry) -> None:
        """Initialize HACS options flow."""
        super().__init__(config_entry)
        self.coordinator: BermudaDataUpdateCoordinator
        self.devices: dict[str, BermudaDevice]
        self._last_ref_power = None
        self._last_device = None
        self._last_scanner = None
        self._last_attenuation = None
        self._last_scanner_info = None

    async def async_step_init(self, user_input=None):  # pylint: disable=unused-argument
        """Manage the options."""
        self.coordinator = self.config_entry.runtime_data.coordinator
        self.devices = self.coordinator.devices

        messages = {}
        active_devices = self.coordinator.count_active_devices()
        active_scanners = self.coordinator.count_active_scanners()

        messages["device_counter_active"] = f"{active_devices}"
        messages["device_counter_devices"] = f"{len(self.devices)}"
        messages["scanner_counter_active"] = f"{active_scanners}"
        messages["scanner_counter_scanners"] = f"{len(self.coordinator.scanner_list)}"

        if len(self.coordinator.scanner_list) == 0:
            messages["status"] = (
                "You need to configure some bluetooth scanners before Bermuda will have anything to work with. "
                "Any one of esphome bluetooth_proxy, Shelly bluetooth proxy or local bluetooth adaptor should get "
                "you started."
            )
        elif active_devices == 0:
            messages["status"] = (
                "No bluetooth devices are actively being reported from your scanners. "
                "You will need to solve this before Bermuda can be of much help."
            )
        else:
            messages["status"] = "You have at least some active devices, this is good."

        # Build a markdown table of scanners so the user can see what's up.
        scanner_table = "\n\nStatus of scanners:\n\n|Scanner|Address|Last advertisement|\n|---|---|---:|\n"
        # Use emoji to indicate if age is "good"
        for scanner in self.coordinator.get_active_scanner_summary():
            age = int(scanner.get("last_stamp_age", 999))
            if age < 2:
                status = '<ha-icon icon="mdi:check-circle-outline"></ha-icon>'
            elif age < 10:
                status = '<ha-icon icon="mdi:alert-outline"></ha-icon>'
            else:
                status = '<ha-icon icon="mdi:skull-crossbones"></ha-icon>'
            # Remove centre octets from mac for condensed, privatised display
            shortmac = mac_redact(scanner.get("address", "ERR"))
            scanner_table += (
                f"| {scanner.get('name', 'NAME_ERR')}| [{shortmac}]"
                f"| {status} {(scanner.get('last_stamp_age', DISTANCE_INFINITE)):.2f} seconds ago.|\n"
            )
        messages["status"] += scanner_table

        # return await self.async_step_globalopts()
        return self.async_show_menu(
            step_id="init",
            menu_options={
                "globalopts": "Global Options",
                "selectdevices": "Select Devices",
                "calibration1_global": "Calibration 1: Global",
                "calibration2_scanners": "Calibration 2: Scanner RSSI Offsets",
                "bulk_import": "Bulk Import Map & Scanners",
                "bulk_export": "Export Map & Scanners (JSON)",
            },
            description_placeholders=messages,
        )

    async def async_step_globalopts(self, user_input=None):
        """Handle global options flow."""
        if user_input is not None:
            self.options.update(user_input)
            return await self._update_options()

        data_schema = {
            vol.Required(
                CONF_MAX_RADIUS,
                default=self.options.get(CONF_MAX_RADIUS, DEFAULT_MAX_RADIUS),
            ): vol.Coerce(float),
            vol.Required(
                CONF_MAX_VELOCITY,
                default=self.options.get(CONF_MAX_VELOCITY, DEFAULT_MAX_VELOCITY),
            ): vol.Coerce(float),
            vol.Required(
                CONF_DEVTRACK_TIMEOUT,
                default=self.options.get(CONF_DEVTRACK_TIMEOUT, DEFAULT_DEVTRACK_TIMEOUT),
            ): vol.Coerce(int),
            vol.Required(
                CONF_UPDATE_INTERVAL,
                default=self.options.get(CONF_UPDATE_INTERVAL, DEFAULT_UPDATE_INTERVAL),
            ): vol.Coerce(float),
            vol.Required(
                CONF_SMOOTHING_SAMPLES,
                default=self.options.get(CONF_SMOOTHING_SAMPLES, DEFAULT_SMOOTHING_SAMPLES),
            ): vol.Coerce(int),
            vol.Required(
                CONF_ATTENUATION,
                default=self.options.get(CONF_ATTENUATION, DEFAULT_ATTENUATION),
            ): vol.Coerce(float),
            vol.Required(
                CONF_REF_POWER,
                default=self.options.get(CONF_REF_POWER, DEFAULT_REF_POWER),
            ): vol.Coerce(float),
        }

        return self.async_show_form(step_id="globalopts", data_schema=vol.Schema(data_schema))

    async def async_step_selectdevices(self, user_input=None):
        """Handle a flow initialized by the user."""
        if user_input is not None:
            self.options.update(user_input)
            return await self._update_options()

        # Grab the co-ordinator's device list so we can build a selector from it.
        self.devices = self.config_entry.runtime_data.coordinator.devices

        # Where we store the options before building the selector
        options_list = []
        options_metadevices = []  # These will be first in the list
        options_otherdevices = []  # These will be last.
        options_randoms = []  # Random MAC addresses - very last!

        for device in self.devices.values():
            # Iterate through all the discovered devices to build the options list

            name = device.name

            if device.is_scanner:
                # We don't "track" scanner devices, per se
                continue
            if device.address_type == ADDR_TYPE_PRIVATE_BLE_DEVICE:
                # Private BLE Devices get configured automagically, skip
                continue
            if device.address_type == ADDR_TYPE_IBEACON:
                # This is an iBeacon meta-device
                if len(device.metadevice_sources) > 0:
                    source_mac = f"[{device.metadevice_sources[0].upper()}]"
                else:
                    source_mac = ""

                options_metadevices.append(
                    SelectOptionDict(
                        value=device.address.upper(),
                        label=f"iBeacon: {device.address.upper()} {source_mac} "
                        f"{name if device.address.upper() != name.upper() else ''}",
                    )
                )
                continue

            if device.address_type == BDADDR_TYPE_RANDOM_RESOLVABLE:
                # This is a random MAC, we should tag it as such

                if device.last_seen < monotonic_time_coarse() - (60 * 60 * 2):  # two hours
                    # A random MAC we haven't seen for a while is not much use, skip
                    continue

                options_randoms.append(
                    SelectOptionDict(
                        value=device.address.upper(),
                        label=f"[{device.address.upper()}] {name} (Random MAC)",
                    )
                )
                continue

            # Default, unremarkable devices, just pop them in the list.
            options_otherdevices.append(
                SelectOptionDict(
                    value=device.address.upper(),
                    label=f"[{device.address.upper()}] {name}",
                )
            )

        # build the final list with "preferred" devices first.
        options_metadevices.sort(key=lambda item: item["label"])
        options_otherdevices.sort(key=lambda item: item["label"])
        options_randoms.sort(key=lambda item: item["label"])
        options_list.extend(options_metadevices)
        options_list.extend(options_otherdevices)
        options_list.extend(options_randoms)

        for address in self.options.get(CONF_DEVICES, []):
            # Now check for any configured devices that weren't discovered, and add them
            if not next(
                (item for item in options_list if item["value"] == address.upper()),
                False,
            ):
                options_list.append(SelectOptionDict(value=address.upper(), label=f"[{address}] (saved)"))

        data_schema = {
            vol.Optional(
                CONF_DEVICES,
                default=self.options.get(CONF_DEVICES, []),
            ): SelectSelector(SelectSelectorConfig(options=options_list, multiple=True)),
        }

        return self.async_show_form(step_id="selectdevices", data_schema=vol.Schema(data_schema))

    async def async_step_calibration1_global(self, user_input=None):
        # FIXME: This is ridiculous. But I can't yet find a better way.
        _ugly_token_hack = {
            # These are work-arounds for (broken?) placeholder substitutions.
            # I've not been able to find out why, but just having <details> in the
            # en.json will cause placeholders to break, due to *something* treating
            # the html elements as placeholders.
            "details": "<details>",
            "details_end": "</details>",
            "summary": "<summary>",
            "summary_end": "</summary>",
        }

        if user_input is not None:
            if user_input[CONF_SAVE_AND_CLOSE]:
                # Update the running options (this propagates to coordinator etc)
                self.options.update(
                    {
                        CONF_ATTENUATION: user_input[CONF_ATTENUATION],
                        CONF_REF_POWER: user_input[CONF_REF_POWER],
                    }
                )
                # Ideally, we'd like to just save out the config entry and return to the main menu.
                # Unfortunately, doing so seems to break the chosen device (for at least 15 seconds or so)
                # until it gets re-invigorated. My guess is that the link between coordinator and the
                # sensor entity might be getting broken, but not entirely sure.
                # For now disabling the return-to-menu and instead we finish out the flow.

                # Previous block for returning to menu:
                # # Let's update the options - but we don't want to call create entry as that will close the flow.
                # # This will save out the config entry:
                # self.hass.config_entries.async_update_entry(self.config_entry, options=self.options)
                # Reset last device so that the next step doesn't think it exists.
                # self._last_device = None
                # return await self.async_step_init()

                # Current block for finishing the flow:
                return await self._update_options()

            self._last_ref_power = user_input[CONF_REF_POWER]
            self._last_attenuation = user_input[CONF_ATTENUATION]
            self._last_device = user_input[CONF_DEVICES]
            self._last_scanner = user_input[CONF_SCANNERS]

        # TODO: Switch this to be a device selector when devices are made for scanners
        scanner_options = [
            SelectOptionDict(
                value=scanner,
                label=self.coordinator.devices[scanner].name if scanner in self.coordinator.devices else scanner,
            )
            for scanner in self.coordinator.scanner_list
        ]
        data_schema = {
            vol.Required(
                CONF_DEVICES,
                default=self._last_device if self._last_device is not None else vol.UNDEFINED,
            ): DeviceSelector(DeviceSelectorConfig(integration=DOMAIN)),
            vol.Required(
                CONF_SCANNERS,
                default=self._last_scanner if self._last_scanner is not None else vol.UNDEFINED,
            ): SelectSelector(
                SelectSelectorConfig(
                    options=scanner_options,
                    multiple=False,
                    mode=SelectSelectorMode.DROPDOWN,
                )
            ),
            vol.Required(
                CONF_REF_POWER,
                default=self._last_ref_power
                if self._last_ref_power is not None
                else self.options.get(CONF_REF_POWER, DEFAULT_REF_POWER),
            ): vol.Coerce(float),
            vol.Required(
                CONF_ATTENUATION,
                default=self._last_attenuation
                if self._last_attenuation is not None
                else self.options.get(CONF_ATTENUATION, DEFAULT_ATTENUATION),
            ): vol.Coerce(float),
            vol.Optional(CONF_SAVE_AND_CLOSE, default=False): vol.Coerce(bool),
        }
        if user_input is None:
            return self.async_show_form(
                step_id="calibration1_global",
                data_schema=vol.Schema(data_schema),
                description_placeholders=_ugly_token_hack
                | {"suffix": "After you click Submit, the new distances will be shown here."},
            )
        results_str = ""
        device = self._get_bermuda_device_from_registry(user_input[CONF_DEVICES])
        if device is not None:
            scanner = device.get_scanner(user_input[CONF_SCANNERS])
            if scanner is None:
                return self.async_show_form(
                    step_id="calibration1_global",
                    errors={"err_scanner_no_record": "The selected scanner hasn't (yet) seen this device."},
                    data_schema=vol.Schema(data_schema),
                    description_placeholders=_ugly_token_hack
                    | {"suffix": "After you click Submit, the new distances will be shown here."},
                )

            distances = [
                rssi_to_metres(historical_rssi, self._last_ref_power, self._last_attenuation)
                for historical_rssi in scanner.hist_rssi
            ]

            # Build a markdown table showing distance and rssi history for the
            # selected device / scanner combination
            results_str = f"| {device.name} |"
            # Limit the number of columns to what's available up to a max of 5.
            cols = min(5, len(distances), len(scanner.hist_rssi))
            for i in range(cols):
                results_str += f" {i} |"
            results_str += "\n|---|"
            for i in range(cols):  # noqa for unused var i
                results_str += "---:|"

            results_str += "\n| Estimate (m) |"
            for i in range(cols):
                results_str += f" `{distances[i]:>5.2f}`|"
            results_str += "\n| RSSI Actual |"
            for i in range(cols):
                results_str += f" `{scanner.hist_rssi[i]:>5}`|"
            results_str += "\n"

        return self.async_show_form(
            step_id="calibration1_global",
            data_schema=vol.Schema(data_schema),
            description_placeholders=_ugly_token_hack
            | {
                "suffix": (
                    f"Recent distances, calculated using `ref_power = {self._last_ref_power}` "
                    f"and `attenuation = {self._last_attenuation}` (values from new...old):\n\n{results_str}"
                ),
            },
        )

    async def async_step_calibration2_scanners(self, user_input=None):
        """
        Per-scanner calibration of rssi_offset.

        Prompts the user to select a configured device, then adjust the offset
        so that the estimated distance to each proxy is correct (typically by
        placing device at 1m from each proxy in turn).

        Distances are recalculated and displayed each time the user presses
        Submit, and they check "Save and Close" to save the config.
        """
        if user_input is not None:
            if user_input[CONF_SAVE_AND_CLOSE]:
                # Convert the name-based dict to use MAC addresses
                rssi_offset_by_address = {}
                for address in self.coordinator.scanner_list:
                    scanner_name = self.coordinator.devices[address].name
                    val = user_input[CONF_SCANNER_INFO][scanner_name]
                    # Clip to keep in sensible range, fixes #497
                    rssi_offset_by_address[address] = max(min(val, 127), -127)

                self.options.update({CONF_RSSI_OFFSETS: rssi_offset_by_address})
                # Per previous step, returning elsewhere in the flow after updating the entry doesn't
                # seem to work, so we'll just save and close the flow.
                # # Let's update the options - but we don't want to call create entry as that will close the flow.
                # self.hass.config_entries.async_update_entry(self.config_entry, options=self.options)
                # # Reset last device so that the next step doesn't think it exists.
                # self._last_device = None
                # self._last_scanner_info = None
                # return await self.async_step_init()

                # Save the config entry and close the flow.
                return await self._update_options()

            # It's a refresh, basically...
            self._last_scanner_info = user_input[CONF_SCANNER_INFO]
            self._last_device = user_input[CONF_DEVICES]

        saved_rssi_offsets = self.options.get(CONF_RSSI_OFFSETS, {})
        rssi_offset_dict = {}

        for scanner in self.coordinator.scanner_list:
            scanner_name = self.coordinator.devices[scanner].name
            rssi_offset_dict[scanner_name] = saved_rssi_offsets.get(scanner, 0)
        data_schema = {
            vol.Required(
                CONF_DEVICES,
                default=self._last_device if self._last_device is not None else vol.UNDEFINED,
            ): DeviceSelector(DeviceSelectorConfig(integration=DOMAIN)),
            vol.Required(
                CONF_SCANNER_INFO,
                default=rssi_offset_dict if not self._last_scanner_info else self._last_scanner_info,
            ): ObjectSelector(),
            vol.Optional(CONF_SAVE_AND_CLOSE, default=False): vol.Coerce(bool),
        }
        if user_input is None:
            return self.async_show_form(
                step_id="calibration2_scanners",
                data_schema=vol.Schema(data_schema),
                description_placeholders={"suffix": "After you click Submit, the new distances will be shown here."},
            )
        if isinstance(self._last_device, str):
            device = self._get_bermuda_device_from_registry(self._last_device)
        results_str = ""
        if device is not None and isinstance(self._last_scanner_info, dict):
            results = {}
            # Gather new estimates for distances using rssi hist and the new offset.
            for scanner in self.coordinator.scanner_list:
                scanner_name = self.coordinator.devices[scanner].name
                cur_offset = self._last_scanner_info.get(scanner_name, 0)
                if (scanneradvert := device.get_scanner(scanner)) is not None:
                    results[scanner_name] = [
                        rssi_to_metres(
                            historical_rssi + cur_offset,
                            self.options.get(CONF_REF_POWER, DEFAULT_REF_POWER),
                            self.options.get(CONF_ATTENUATION, DEFAULT_ATTENUATION),
                        )
                        for historical_rssi in scanneradvert.hist_rssi
                    ]
            # Format the results for display (HA has full markdown support!)
            results_str = "| Scanner | 0 | 1 | 2 | 3 | 4 |\n|---|---:|---:|---:|---:|---:|"
            for scanner_name, distances in results.items():
                results_str += f"\n|{scanner_name}|"
                for i in range(5):
                    # We round to 2 places (1cm) and pad to fit nn.nn
                    try:
                        results_str += f" `{distances[i]:>6.2f}`|"
                    except IndexError:
                        results_str += "`-`|"
            results_str += "\n\n"

        return self.async_show_form(
            step_id="calibration2_scanners",
            data_schema=vol.Schema(data_schema),
            description_placeholders={"suffix": results_str},
        )

    def _get_bermuda_device_from_registry(self, registry_id: str) -> BermudaDevice | None:
        """
        Given a device registry device id, return the associated MAC address.

        Returns None if the id can not be resolved to a mac.
        """
        devreg = dr.async_get(self.hass)
        device = devreg.async_get(registry_id)
        device_address = None
        if device is not None:
            for connection in device.connections:
                if connection[0] in {
                    DOMAIN_PRIVATE_BLE_DEVICE,
                    dr.CONNECTION_BLUETOOTH,
                    "ibeacon",
                }:
                    device_address = connection[1]
                    break
            if device_address is not None:
                return self.coordinator.devices[device_address.lower()]
        # We couldn't match the HA device id to a bermuda device mac.
        return None

    async def async_step_bulk_import(self, user_input=None):
        """Handle JSON bulk import of scanner positions, floors, and rooms."""
        from .util import mac_norm

        # Validation schemas
        FLOOR_SCHEMA = vol.Schema({
            vol.Required("id"): str,
            vol.Required("name"): str,
            vol.Required("bounds"): vol.All(
                list,
                vol.Length(min=2, max=2),
                [vol.All(list, vol.Length(min=3, max=3), [vol.Coerce(float)])],
            ),
        })

        ROOM_SCHEMA = vol.Schema({
            vol.Required("id"): str,
            vol.Required("name"): str,
            vol.Required("floor"): str,
            vol.Required("area_id"): str,
            vol.Required("points"): vol.All(
                list,
                vol.Length(min=3),  # Polygon needs at least 3 points
                [vol.All(list, vol.Length(min=2, max=2), [vol.Coerce(float)])],
            ),
        })

        NODE_SCHEMA = vol.Schema({
            vol.Required("id"): str,  # MAC address
            vol.Optional("name"): str,
            vol.Required("point"): vol.All(
                list,
                vol.Length(min=3, max=3),
                [vol.Coerce(float)],
            ),
        })

        BULK_IMPORT_SCHEMA = vol.Schema({
            vol.Optional("floors", default=[]): [FLOOR_SCHEMA],
            vol.Optional("rooms", default=[]): [ROOM_SCHEMA],
            vol.Optional("nodes", default=[]): [NODE_SCHEMA],
        })

        if user_input is not None:
            # Validate and import
            try:
                data = json.loads(user_input[CONF_JSON_IMPORT])
                # Validate against BULK_IMPORT_SCHEMA
                validated_data = BULK_IMPORT_SCHEMA(data)

                # Additional validation
                errors = self._validate_bulk_import(validated_data)

                if errors:
                    return self.async_show_form(
                        step_id="bulk_import",
                        data_schema=self._build_bulk_import_schema(user_input[CONF_IMPORT_MODE]),
                        errors=errors,
                        description_placeholders=self._build_import_description(),
                    )

                # Store in config entry
                await self._apply_bulk_import(validated_data, user_input[CONF_IMPORT_MODE])

                return await self._update_options()

            except json.JSONDecodeError as e:
                return self.async_show_form(
                    step_id="bulk_import",
                    data_schema=self._build_bulk_import_schema(),
                    errors={"base": f"Invalid JSON: {e}"},
                    description_placeholders=self._build_import_description(),
                )
            except vol.Invalid as e:
                return self.async_show_form(
                    step_id="bulk_import",
                    data_schema=self._build_bulk_import_schema(),
                    errors={"base": f"Validation error: {e}"},
                    description_placeholders=self._build_import_description(),
                )

        # Show form
        return self.async_show_form(
            step_id="bulk_import",
            data_schema=self._build_bulk_import_schema(),
            description_placeholders=self._build_import_description(),
        )

    def _build_bulk_import_schema(self, default_mode="replace"):
        """Build schema for bulk import form."""
        return vol.Schema({
            vol.Required(CONF_JSON_IMPORT): TextSelector(
                TextSelectorConfig(
                    type=TextSelectorType.TEXT,
                    multiline=True,
                )
            ),
            vol.Required(CONF_IMPORT_MODE, default=default_mode): SelectSelector(
                SelectSelectorConfig(
                    options=[
                        SelectOptionDict(value="replace", label="Replace All"),
                        SelectOptionDict(value="merge", label="Merge (keep existing)"),
                    ],
                    mode=SelectSelectorMode.DROPDOWN,
                )
            ),
        })

    def _build_import_description(self):
        """Build description placeholders for bulk import form."""
        return {
            "instructions": (
                "Paste JSON configuration for scanner positions, floors, and rooms.\n\n"
                "**Import Mode:**\n"
                "- **Replace All**: Removes existing data and imports new\n"
                "- **Merge**: Keeps existing data, updates matching IDs\n\n"
                "See [scanner_positions.json.example](https://github.com/agittins/bermuda/blob/main/scanner_positions.json.example) for format."
            )
        }

    def _validate_bulk_import(self, data: dict) -> dict[str, str] | None:
        """Validate bulk import data. Returns error dict or None."""
        from .util import mac_norm

        errors = {}

        # Check floor references in rooms
        floor_ids = {f["id"] for f in data.get("floors", [])}
        for room in data.get("rooms", []):
            if room["floor"] not in floor_ids:
                errors["base"] = f"Room '{room['id']}' references unknown floor '{room['floor']}'"
                return errors

        # Check for duplicate IDs
        floor_id_list = [f["id"] for f in data.get("floors", [])]
        if len(floor_id_list) != len(set(floor_id_list)):
            errors["base"] = "Duplicate floor IDs found"
            return errors

        room_id_list = [r["id"] for r in data.get("rooms", [])]
        if len(room_id_list) != len(set(room_id_list)):
            errors["base"] = "Duplicate room IDs found"
            return errors

        # Normalize and check scanner MACs
        scanner_macs = []
        for node in data.get("nodes", []):
            try:
                normalized = mac_norm(node["id"])
                scanner_macs.append(normalized)
            except (KeyError, TypeError, ValueError):
                errors["base"] = f"Invalid MAC address format: {node['id']}"
                return errors

        if len(scanner_macs) != len(set(scanner_macs)):
            errors["base"] = "Duplicate scanner MAC addresses found"
            return errors

        return None

    async def _apply_bulk_import(self, data: dict, mode: str):
        """Apply validated bulk import data to config entry."""
        from .util import mac_norm

        if mode == "replace":
            # Replace all data
            self.options[CONFDATA_FLOORS] = data.get("floors", [])
            self.options[CONFDATA_ROOMS] = data.get("rooms", [])

            # Scanner positions as dict keyed by normalized MAC
            positions = {}
            for node in data.get("nodes", []):
                mac = mac_norm(node["id"])
                positions[mac] = {
                    "name": node.get("name"),
                    "point": node["point"],
                }
            self.options[CONFDATA_SCANNER_POSITIONS] = positions

        else:  # mode == "merge"
            # Merge floors (update existing, add new)
            existing_floors = {f["id"]: f for f in self.options.get(CONFDATA_FLOORS, [])}
            for floor in data.get("floors", []):
                existing_floors[floor["id"]] = floor
            self.options[CONFDATA_FLOORS] = list(existing_floors.values())

            # Same for rooms
            existing_rooms = {r["id"]: r for r in self.options.get(CONFDATA_ROOMS, [])}
            for room in data.get("rooms", []):
                existing_rooms[room["id"]] = room
            self.options[CONFDATA_ROOMS] = list(existing_rooms.values())

            # Same for scanner positions
            existing_positions = self.options.get(CONFDATA_SCANNER_POSITIONS, {})
            for node in data.get("nodes", []):
                mac = mac_norm(node["id"])
                existing_positions[mac] = {
                    "name": node.get("name"),
                    "point": node["point"],
                }
            self.options[CONFDATA_SCANNER_POSITIONS] = existing_positions

    async def async_step_bulk_export(self, user_input=None):
        """Export current scanner positions as JSON."""
        # Build JSON from current config entry + live data
        export_data = {
            "floors": self.options.get(CONFDATA_FLOORS, []),
            "rooms": self.options.get(CONFDATA_ROOMS, []),
            "nodes": []
        }

        # Add scanner positions from config entry or live devices
        positions = self.options.get(CONFDATA_SCANNER_POSITIONS, {})
        for mac, pos_data in positions.items():
            export_data["nodes"].append({
                "id": mac.upper(),
                "name": pos_data.get("name", ""),
                "point": pos_data["point"],
            })

        # Also include scanners with positions not in config entry
        for device in self.coordinator.devices.values():
            if device.is_scanner and device.position:
                mac = device.address
                if mac not in positions:
                    export_data["nodes"].append({
                        "id": mac.upper(),
                        "name": device.name,
                        "point": list(device.position),
                    })

        json_output = json.dumps(export_data, indent=2)

        # Show form with readonly text area
        return self.async_show_form(
            step_id="bulk_export",
            data_schema=vol.Schema({}),  # Empty schema, just show info
            description_placeholders={
                "json_output": f"```json\n{json_output}\n```",
                "instructions": "Copy the JSON below to save your current configuration.",
            },
        )

    async def _update_options(self):
        """Update config entry options."""
        return self.async_create_entry(title=NAME, data=self.options)

