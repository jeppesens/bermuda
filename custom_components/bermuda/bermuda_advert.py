"""
Bermuda's internal representation of a device to scanner relationship.

This can also be thought of as the representation of an advertisement
received by a given scanner, in that it's the advert that links the
device to a scanner. Multiple scanners will receive a given advert, but
each receiver experiences it (well, the rssi) uniquely.

Every bluetooth scanner is a BermudaDevice, but this class
is the nested entry that gets attached to each device's `scanners`
dict. It is a sub-set of a 'device' and will have attributes specific
to the combination of the scanner and the device it is reporting.
"""

from __future__ import annotations

import statistics
from collections import deque
from itertools import islice
from typing import TYPE_CHECKING
from typing import Final

from bluetooth_data_tools import monotonic_time_coarse

from .const import _LOGGER
from .const import _LOGGER_SPAM_LESS
from .const import CONF_ATTENUATION
from .const import CONF_MAX_VELOCITY
from .const import CONF_REF_POWER
from .const import CONF_RSSI_OFFSETS
from .const import CONF_SMOOTHING_SAMPLES
from .const import DISTANCE_INFINITE
from .const import DISTANCE_TIMEOUT
from .const import HIST_KEEP_COUNT
from .util import clean_charbuf
from .util import rssi_to_metres

if TYPE_CHECKING:
    from bleak.backends.scanner import AdvertisementData

    from .bermuda_device import BermudaDevice

# The if instead of min/max triggers PLR1730, but when
# split over two lines, ruff removes it, then complains again.
# so we're just disabling it for the whole file.
# https://github.com/astral-sh/ruff/issues/4244
# ruff: noqa: PLR1730


class AdaptivePercentileRSSI:
    """
    Adaptive percentile-based RSSI filtering using IQR (Interquartile Range) method.

    This filtering approach is inspired by ESPresense's implementation and provides
    robust outlier rejection using Tukey's outlier detection method. It maintains
    a time-windowed buffer of RSSI readings and adaptively sizes the buffer based
    on advertisement rate.

    The IQR method calculates quartiles (Q1, median, Q3) and applies a Tukey fence
    to reject outliers: [Q1 - k*IQR, Q3 + k*IQR]. Values within the fence are
    averaged to produce a filtered RSSI value.
    """

    def __init__(
        self,
        time_window_ms: float = 15000,
        initial_max_readings: int = 20,
        min_readings: int = 10,
        max_readings: int = 200,
        iqr_coefficient: float = 1.5,
    ) -> None:
        """
        Initialize adaptive percentile RSSI filter.

        Args:
            time_window_ms: Time window in milliseconds for keeping readings (default 15000ms = 15s)
            initial_max_readings: Initial buffer size (default 20)
            min_readings: Minimum buffer size (default 10)
            max_readings: Maximum buffer size (default 200)
            iqr_coefficient: Tukey fence coefficient k (default 1.5, standard for outlier detection)
        """
        self.time_window_ms = time_window_ms
        self.min_readings = min_readings
        self.max_readings = max_readings
        self.iqr_coefficient = iqr_coefficient
        # Store tuples of (rssi, timestamp)
        self.readings: deque[tuple[float, float]] = deque(maxlen=initial_max_readings)
        self._last_adjustment_time: float = 0
        self._adjustment_interval: float = 10.0  # Adjust buffer size every 10 seconds

    def add_measurement(self, rssi: float, timestamp: float) -> None:
        """
        Add a new RSSI measurement with timestamp.

        Args:
            rssi: RSSI value in dBm
            timestamp: Timestamp of the measurement (monotonic time)
        """
        self.readings.append((rssi, timestamp))
        self._remove_expired(timestamp)
        self._adjust_buffer_size(timestamp)

    def _remove_expired(self, current_time: float) -> None:
        """Remove readings older than the time window."""
        cutoff_time = current_time - (self.time_window_ms / 1000.0)
        # Remove from left (oldest) while they're too old
        while self.readings and self.readings[0][1] < cutoff_time:
            self.readings.popleft()

    def _adjust_buffer_size(self, current_time: float) -> None:
        """
        Dynamically adjust buffer size based on advertisement rate.

        This ensures we maintain consistent time-window coverage regardless of
        whether the device advertises every 100ms or 1000ms.
        """
        # Only adjust periodically to avoid excessive resizing
        if current_time - self._last_adjustment_time < self._adjustment_interval:
            return

        self._last_adjustment_time = current_time

        if len(self.readings) < 2:
            return

        # Calculate average interval between advertisements
        time_span = current_time - self.readings[0][1]
        avg_interval = time_span / len(self.readings) if len(self.readings) > 0 else 1.0

        # Calculate target buffer size to fill time window
        # Add small epsilon to avoid division by zero
        target_size = int(self.time_window_ms / (max(avg_interval, 0.001) * 1000))
        target_size = max(self.min_readings, min(target_size, self.max_readings))

        # Resize deque if needed
        if target_size != self.readings.maxlen:
            _LOGGER.debug(
                "Adjusting RSSI filter buffer size from %d to %d (avg interval: %.3fs)",
                self.readings.maxlen,
                target_size,
                avg_interval,
            )
            # Create new deque with new maxlen, preserving most recent readings
            new_readings: deque[tuple[float, float]] = deque(
                islice(self.readings, max(0, len(self.readings) - target_size), len(self.readings)),
                maxlen=target_size,
            )
            self.readings = new_readings

    def _calculate_percentile(self, sorted_values: list[float], percentile: float) -> float:
        """
        Calculate percentile from sorted values using linear interpolation.

        Args:
            sorted_values: List of values sorted in ascending order
            percentile: Percentile to calculate (0.0 to 1.0)

        Returns:
            Interpolated percentile value
        """
        if not sorted_values:
            return 0.0
        if len(sorted_values) == 1:
            return sorted_values[0]

        # Use linear interpolation between closest ranks
        index = percentile * (len(sorted_values) - 1)
        lower_idx = int(index)
        upper_idx = min(lower_idx + 1, len(sorted_values) - 1)
        fraction = index - lower_idx

        return sorted_values[lower_idx] * (1 - fraction) + sorted_values[upper_idx] * fraction

    def get_median_iqr(self, k: float | None = None) -> float | None:
        """
        Get mean of RSSI values within Tukey fence (Q1 - k*IQR, Q3 + k*IQR).

        This implements Tukey's outlier detection method:
        1. Calculate Q1 (25th percentile), median (50th), Q3 (75th percentile)
        2. Calculate IQR = Q3 - Q1
        3. Define fence: [Q1 - k*IQR, Q3 + k*IQR]
        4. Return mean of values within fence, or median if all rejected

        Args:
            k: Tukey fence coefficient (default: use self.iqr_coefficient, typically 1.5)

        Returns:
            Filtered RSSI value, or None if no readings available
        """
        if not self.readings:
            return None

        if len(self.readings) == 1:
            return self.readings[0][0]

        k = k if k is not None else self.iqr_coefficient

        # Extract and sort RSSI values
        rssi_values = sorted([r[0] for r in self.readings])

        # Calculate quartiles
        q1 = self._calculate_percentile(rssi_values, 0.25)
        q2_median = self._calculate_percentile(rssi_values, 0.50)
        q3 = self._calculate_percentile(rssi_values, 0.75)
        iqr = q3 - q1

        # Calculate Tukey fence bounds
        lower_fence = q1 - k * iqr
        upper_fence = q3 + k * iqr

        # Filter values within fence
        survivors = [v for v in rssi_values if lower_fence <= v <= upper_fence]

        # Return mean of survivors, or median if all were rejected
        if survivors:
            return statistics.mean(survivors)
        return q2_median

    def get_rssi_variance(self) -> float:
        """
        Calculate variance of RSSI values in the buffer.

        Returns:
            Variance of RSSI values, or 0.0 if insufficient data
        """
        if len(self.readings) < 2:
            return 0.0

        rssi_values = [r[0] for r in self.readings]
        return statistics.variance(rssi_values)

    def get_distance_variance(self, ref_power: float, attenuation: float) -> float:
        """
        Calculate variance of distance estimates from RSSI variance.

        Converts RSSI readings to distances using the log-distance path loss model,
        then calculates the variance of those distance values.

        Args:
            ref_power: Reference RSSI at 1 meter (typically -65 to -75 dBm)
            attenuation: Path loss exponent (typically 2.7 to 3.5)

        Returns:
            Variance of distance estimates in meters, or 0.0 if insufficient data
        """
        if len(self.readings) < 2:
            return 0.0

        # Convert each RSSI to distance
        distances = [10 ** ((ref_power - rssi) / (10 * attenuation)) for rssi, _ in self.readings]

        return statistics.variance(distances)

    def get_sample_count(self) -> int:
        """Get current number of samples in the buffer."""
        return len(self.readings)

    def clear(self) -> None:
        """Clear all readings from the buffer."""
        self.readings.clear()
        self._last_adjustment_time = 0


class BermudaAdvert(dict):
    """
    Represents details from a scanner relevant to a specific device.

    Effectively a link between two BermudaDevices, being the tracked device
    and the scanner device. So each transmitting device will have a collection
    of these BermudaDeviceScanner entries, one for each scanner that has picked
    up the advertisement.

    This is created (and updated) by the receipt of an advertisement, which represents
    a BermudaDevice hearing an advert from another BermudaDevice, if that makes sense!

    A BermudaDevice's "adverts" property will contain one of these for each
    scanner that has "seen" it.

    """

    def __hash__(self) -> int:
        """The device-mac / scanner mac uniquely identifies a received advertisement pair."""
        return hash((self.device_address, self.scanner_address))

    def __init__(
        self,
        parent_device: BermudaDevice,  # The device being tracked
        advertisementdata: AdvertisementData,  # The advertisement info from the device, received by the scanner
        options,
        scanner_device: BermudaDevice,  # The scanner device that "saw" it.
    ) -> None:
        self.scanner_address: Final[str] = scanner_device.address
        self.device_address: Final[str] = parent_device.address
        self._device = parent_device
        self.ref_power: float = self._device.ref_power  # Take from parent at first, might be changed by metadevice l8r
        self.apply_new_scanner(scanner_device)

        self.options = options

        self.stamp: float = 0
        self.new_stamp: float | None = None  # Set when a new advert is loaded from update
        self.rssi: float | None = None
        self.tx_power: float | None = None
        self.rssi_distance: float | None = None
        self.rssi_distance_raw: float
        self.stale_update_count = 0  # How many times we did an update but no new stamps were found.

        # Adaptive percentile RSSI filter for advanced outlier rejection
        self.adaptive_rssi_filter: AdaptivePercentileRSSI = AdaptivePercentileRSSI()
        self.rssi_filtered: float | None = None  # Filtered RSSI from IQR method
        self.rssi_variance: float = 0.0  # Variance of RSSI measurements
        self.distance_variance: float = 0.0  # Variance of distance estimates

        # Using deques for O(1) prepend operations in rolling history buffers
        self.hist_stamp: deque[float] = deque(maxlen=HIST_KEEP_COUNT)
        self.hist_rssi: deque[int] = deque(maxlen=HIST_KEEP_COUNT)
        self.hist_distance: deque[float] = deque(maxlen=HIST_KEEP_COUNT)
        self.hist_distance_by_interval: deque[float] = deque(maxlen=HIST_KEEP_COUNT)  # updated per-interval
        self.hist_interval: deque = deque(maxlen=HIST_KEEP_COUNT)  # WARNING: This is actually "age of ad when we polled"
        self.hist_velocity: deque[float] = deque(maxlen=HIST_KEEP_COUNT)  # Effective velocity versus previous stamped reading
        self.conf_rssi_offset = self.options.get(CONF_RSSI_OFFSETS, {}).get(self.scanner_address, 0)
        self.conf_ref_power = self.options.get(CONF_REF_POWER)
        self.conf_attenuation = self.options.get(CONF_ATTENUATION)
        self.conf_max_velocity = self.options.get(CONF_MAX_VELOCITY)
        self.conf_smoothing_samples = self.options.get(CONF_SMOOTHING_SAMPLES)
        self.local_name: list[tuple[str, bytes]] = []
        self.manufacturer_data: list[dict[int, bytes]] = []
        self.service_data: list[dict[str, bytes]] = []
        self.service_uuids: list[str] = []

        # Just pass the rest on to update...
        self.update_advertisement(advertisementdata, self.scanner_device)

    def apply_new_scanner(self, scanner_device: BermudaDevice):
        self.name: str = scanner_device.name  # or scandata.scanner.name
        self.scanner_device = scanner_device  # links to the source device
        if self.scanner_address != scanner_device.address:
            _LOGGER.error("Advert %s received new scanner with wrong address %s", self.__repr__(), scanner_device)
        self.area_id: str | None = scanner_device.area_id
        self.area_name: str | None = scanner_device.area_name
        # Only remote scanners log timestamps, local usb adaptors do not.
        self.scanner_sends_stamps = scanner_device.is_remote_scanner

    def update_advertisement(self, advertisementdata: AdvertisementData, scanner_device: BermudaDevice):
        """
        Update gets called every time we see a new packet or
        every time we do a polled update.

        This method needs to update all the history and tracking data for this
        device+scanner combination. This method only gets called when a given scanner
        claims to have data.
        """
        #
        # We might get called without there being a new advert to process, so
        # exit quickly if that's the case (ideally we will catch it earlier in future)
        #
        if scanner_device is not self.scanner_device:
            _LOGGER.debug(
                "Replacing stale scanner device %s with %s", self.scanner_device.__repr__(), scanner_device.__repr__()
            )
            self.apply_new_scanner(scanner_device)

        scanner = self.scanner_device
        new_stamp: float | None = None

        if self.scanner_sends_stamps:
            new_stamp = scanner.async_as_scanner_get_stamp(self.device_address)

            if new_stamp is None:
                self.stale_update_count += 1
                _LOGGER_SPAM_LESS.warning(
                    f"{scanner.name}_{self._device.name}_lacks_stamp",
                    "Advert from %s for %s lacks stamp (scanner may not support timestamps)",
                    scanner.name,
                    self._device.name,
                )
                return

            if self.stamp > new_stamp:
                # The existing stamp is NEWER, bail but complain on the way.
                self.stale_update_count += 1
                _LOGGER.debug("Advert from %s for %s is OLDER than last recorded", scanner.name, self._device.name)
                return

            if self.stamp == new_stamp:
                # We've seen this stamp before. Bail.
                self.stale_update_count += 1
                return

        elif self.rssi != advertisementdata.rssi:
            # If the rssi has changed from last time, consider it "new". Since this scanner does
            # not send stamps, this is probably a USB bluetooth adaptor.
            new_stamp = monotonic_time_coarse() - 3.0  # age usb adaptors slightly, since they are not "fresh"
        else:
            # USB Adaptor has nothing new for us, bail.
            return

        # Update our parent scanner's last_seen if we have a new stamp.
        if new_stamp > self.scanner_device.last_seen + 0.01:  # some slight warp seems common.
            _LOGGER.debug(
                "Advert from %s for %s is %.6fs NEWER than scanner's last_seen, odd",
                self.scanner_device.name,
                self._device.name,
                new_stamp - self.scanner_device.last_seen,
            )
            self.scanner_device.last_seen = new_stamp

        if len(self.hist_stamp) == 0 or new_stamp is not None:
            # this is the first entry or a new one, bring in the new reading
            # and calculate the distance.

            self.rssi = advertisementdata.rssi
            self.hist_rssi.appendleft(self.rssi)

            # Add to adaptive filter and update filtered RSSI
            if new_stamp is not None:
                self.adaptive_rssi_filter.add_measurement(self.rssi, new_stamp)
                self.rssi_filtered = self.adaptive_rssi_filter.get_median_iqr()
                self.rssi_variance = self.adaptive_rssi_filter.get_rssi_variance()
                self.distance_variance = self.adaptive_rssi_filter.get_distance_variance(
                    self.ref_power, self.conf_attenuation
                )

            self._update_raw_distance(reading_is_new=True)

            # Note: this is not actually the interval between adverts,
            # but rather a function of our UPDATE_INTERVAL plus the packet
            # interval. The bluetooth integration does not currently store
            # interval data, only stamps of the most recent packet.
            # So it more accurately reflects "How much time passed between
            # the two last packets we observed" - which should be a multiple
            # of the true inter-packet interval. For stamps from local bluetooth
            # adaptors (usb dongles) it reflects "Which update cycle last saw a
            # different rssi", which will be a multiple of our update interval.
            if new_stamp is not None and self.stamp is not None:
                _interval = new_stamp - self.stamp
            else:
                _interval = None
            self.hist_interval.appendleft(_interval)

            self.stamp = new_stamp or 0
            self.hist_stamp.appendleft(self.stamp)

        # if self.tx_power is not None and scandata.advertisement.tx_power != self.tx_power:
        #     # Not really an erorr, we just don't account for this happening -
        #     # I want to know if it does.
        #     # AJG 2024-01-11: This does happen. Looks like maybe apple devices?
        #     # Changing from warning to debug to quiet users' logs.
        #     # Also happens with esphome set with long beacon interval tx, as it alternates
        #     # between sending some generic advert and the iBeacon advert. ie, it's bogus for that
        #     # case.
        #     _LOGGER.debug(
        #         "Device changed TX-POWER! That was unexpected: %s %sdB",
        #         self.parent_device_address,
        #         scandata.advertisement.tx_power,
        #     )
        self.tx_power = advertisementdata.tx_power

        # Store each of the extra advertisement fields in historical lists.
        # Track if we should tell the parent device to update its name
        _want_name_update = False
        if advertisementdata.local_name is not None:
            # It's not uncommon to find BT devices with nonascii junk in their
            # local_name (like nulls, \n, etc). Store a cleaned version as str
            # and the original as bytes.
            # Devices may also advert multiple names over time.
            nametuplet = (clean_charbuf(advertisementdata.local_name), advertisementdata.local_name.encode())
            if len(self.local_name) == 0 or self.local_name[0] != nametuplet:
                self.local_name.insert(0, nametuplet)
                del self.local_name[HIST_KEEP_COUNT:]
                # Lets see if we should pass the new name up to the parent device.
                if self._device.name_bt_local_name is None or len(self._device.name_bt_local_name) < len(nametuplet[0]):
                    self._device.name_bt_local_name = nametuplet[0]
                    _want_name_update = True

        if len(self.manufacturer_data) == 0 or self.manufacturer_data[0] != advertisementdata.manufacturer_data:
            self.manufacturer_data.insert(0, advertisementdata.manufacturer_data)

            # If manufacturing data changes, we call the update. This is because iBeacons might change their
            # sent details, in which case we need to re-match them.
            self._device.process_manufacturer_data(self)
            _want_name_update = True
            del self.manufacturer_data[HIST_KEEP_COUNT:]

        if len(self.service_data) == 0 or self.service_data[0] != advertisementdata.service_data:
            self.service_data.insert(0, advertisementdata.service_data)
            if advertisementdata.service_data not in self.manufacturer_data[1:]:
                _want_name_update = True
            del self.service_data[HIST_KEEP_COUNT:]

        for service_uuid in advertisementdata.service_uuids:
            if service_uuid not in self.service_uuids:
                self.service_uuids.insert(0, service_uuid)
                _want_name_update = True
                del self.service_uuids[HIST_KEEP_COUNT:]

        if _want_name_update:
            self._device.make_name()

        # Finally, save the new advert timestamp.
        self.new_stamp = new_stamp

    def _update_raw_distance(self, reading_is_new=True) -> float:
        """
        Converts rssi to raw distance and updates history stack and
        returns the new raw distance.

        reading_is_new should only be called by the regular update
        cycle, as it creates a new entry in the histories. Call with
        false if you just need to set / override distance measurements
        immediately, perhaps between cycles, in order to reflect a
        setting change (such as altering a device's ref_power setting).
        """
        # Check if we should use a device-based ref_power
        if not self.ref_power:  # No user-supplied per-device value
            # use global default
            ref_power = self.conf_ref_power
            
            # Warn if ref_power seems unrealistic (likely misconfigured)
            if ref_power is not None and (ref_power > -50 or ref_power < -85):
                _LOGGER_SPAM_LESS.warning(
                    f"ref_power_unusual_{self._device.address}",
                    "Unusual ref_power value %.1f dBm for %s. Typical range is -59 to -75 dBm. "
                    "This may cause inaccurate distance calculations. Please calibrate by measuring "
                    "actual RSSI at 1 meter from device.",
                    ref_power,
                    self._device.name,
                )
        else:
            ref_power = self.ref_power

        distance = rssi_to_metres(self.rssi + self.conf_rssi_offset, ref_power, self.conf_attenuation)
        self.rssi_distance_raw = distance
        if reading_is_new:
            # Add a new historical reading
            self.hist_distance.appendleft(distance)
            # don't insert into hist_distance_by_interval, that's done by the caller.
        elif self.rssi_distance is not None:
            # We are over-riding readings between cycles.
            # We will force the new measurement, but only if we were
            # already showing a "current" distance, as we don't want
            # to "freshen" a measurement that was already out of date,
            # hence the elif not none above.
            self.rssi_distance = distance
            if len(self.hist_distance) > 0:
                self.hist_distance[0] = distance
            else:
                self.hist_distance.append(distance)
            if len(self.hist_distance_by_interval) > 0:
                self.hist_distance_by_interval[0] = distance
            # We don't else because we don't want to *add* a hist-by-interval reading, only
            # modify in-place.
        return distance

    def set_ref_power(self, value: float) -> float | None:
        """
        Set a new reference power and return the resulting distance.

        Typically called from the parent device when either the user changes the calibration
        of ref_power for a device, or when a metadevice takes on a new source device, and
        propagates its own ref_power to our parent.

        Note that it is unlikely to return None as its only returning the raw, not filtered
        distance = the exception being uninitialised entries.
        """
        # When the user updates the ref_power we want to reflect that change immediately,
        # and not subject it to the normal smoothing algo.
        # But make sure it's actually different, in case it's just a metadevice propagating
        # its own ref_power without need.
        if value != self.ref_power:
            self.ref_power = value
            return self._update_raw_distance(False)
        return self.rssi_distance_raw

    def calculate_data(self):
        """
        Filter and update distance estimates.

        All smoothing and noise-management of the distance between a scanner
        and a device should be done in this method, as it is
        guaranteed to be called on every update cycle, for every
        scanner that has ever reported an advert for this device
        (even if it is not reporting one currently).

        If new_stamp is None it implies that the scanner has not reported
        an updated advertisement since our last update cycle,
        so we may need to check if this device should be timed
        out or otherwise dealt with.

        If new_stamp is not None it means we just had an updated
        rssi_distance_raw value which should be processed.

        This is called by self.update, but should also be called for
        any remaining scanners that have not sent in an update in this
        cycle. This is mainly beacuse usb/bluez adaptors seem to flush
        their advertisement lists quicker than we time out, so we need
        to make sure we still update the scanner entry even if the scanner
        no longer carries advert history for this device.

        Note: Noise in RSSI readings is VERY asymmetric. Ultimately,
        a closer distance is *always* more accurate than a previous
        more distant measurement. Any measurement might be true,
        or it is likely longer than the truth - and (almost) never
        shorter.

        For a new, long measurement to be true, we'd want to see some
        indication of rising measurements preceding it, or at least a
        long time since our last measurement.

        It's tempting to treat no recent measurement as implying an increase
        in distance, but doing so would wreak havoc when we later try to
        implement trilateration, so better to simply cut a sensor off as
        "away" from a scanner when it hears no new adverts. DISTANCE_TIMEOUT
        is how we decide how long to wait, and should accommodate for dropped
        packets and for temporary occlusion (dogs' bodies etc)
        """
        new_stamp = self.new_stamp  # should have been set by update()
        self.new_stamp = None  # Clear so we know if an update is missed next cycle

        if self.rssi_distance is None and new_stamp is not None:
            # DEVICE HAS ARRIVED!
            # We have just newly come into range (or we're starting up)
            # accept the new reading as-is.
            self.rssi_distance = self.rssi_distance_raw
            # And ensure the smoothing history gets a fresh start

            if self.rssi_distance_raw is not None:
                # clear tends to be more efficient than re-creating
                # and might have fewer side-effects.
                self.hist_distance_by_interval.clear()
                self.hist_distance_by_interval.append(self.rssi_distance_raw)

        elif new_stamp is None and (self.stamp is None or self.stamp < monotonic_time_coarse() - DISTANCE_TIMEOUT):
            # DEVICE IS AWAY!
            # Last distance reading is stale, mark device distance as unknown.
            self.rssi_distance = None
            # Clear the smoothing history
            if len(self.hist_distance_by_interval) > 0:
                self.hist_distance_by_interval.clear()

        else:
            # Add the current reading (whether new or old) to
            # a historical log that is evenly spaced by update_interval.

            # Verify the new reading is vaguely sensible. If it isn't, we
            # ignore it by duplicating the last cycle's reading.
            if len(self.hist_stamp) > 1:
                # How far (away) did it travel in how long?
                # we check this reading against the recent readings to find
                # the peak average velocity we are alleged to have reached.
                velo_newdistance = self.hist_distance[0]
                velo_newstamp = self.hist_stamp[0]
                peak_velocity = 0
                # walk through the history of distances/stamps, and find
                # the peak
                delta_t = velo_newstamp - self.hist_stamp[1]
                delta_d = velo_newdistance - self.hist_distance[1]
                if delta_t > 0:
                    peak_velocity = delta_d / delta_t
                # if our initial reading is an approach, we are done here
                if peak_velocity >= 0:
                    # Use islice for deque slicing (deques don't support [2:] notation)
                    for old_distance, old_stamp in zip(islice(self.hist_distance, 2, None), islice(self.hist_stamp, 2, None), strict=False):
                        if old_stamp is None:
                            continue  # Skip this iteration if hist_stamp[i] is None

                        delta_t = velo_newstamp - old_stamp
                        if delta_t <= 0:
                            # Additionally, skip if delta_t is zero or negative
                            # to avoid division by zero
                            continue
                        delta_d = velo_newdistance - old_distance

                        velocity = delta_d / delta_t

                        # Don't use max() as it's slower.
                        if velocity > peak_velocity:  # noqa: RUF100, PLR1730
                            # but on subsequent comparisons we only care if they're faster retreats
                            peak_velocity = velocity
                # we've been through the history and have peak velo retreat, or the most recent
                # approach velo.
                velocity = peak_velocity
            else:
                # There's no history, so no velocity
                velocity = 0

            self.hist_velocity.appendleft(velocity)

            if velocity > self.conf_max_velocity:
                if self._device.create_sensor:
                    _LOGGER.debug(
                        "This sparrow %s flies too fast (%2fm/s), ignoring",
                        self._device.name,
                        velocity,
                    )

                # Discard the bogus reading by duplicating the last
                if len(self.hist_distance_by_interval) > 0:
                    self.hist_distance_by_interval.appendleft(self.hist_distance_by_interval[0])
                else:
                    # If nothing to duplicate, just plug in the raw distance.
                    self.hist_distance_by_interval.appendleft(self.rssi_distance_raw)
            else:
                self.hist_distance_by_interval.appendleft(self.rssi_distance_raw)

            # No need to trim - deque maxlen handles this automatically

            # Calculate a moving-window average, that only includes
            # historical values if they're "closer" (ie more reliable).
            #
            # This might be improved by weighting the values by age, but
            # already does a fairly reasonable job of hugging the bottom
            # of the noisy rssi data. A better way to control the maximum
            # slope angle (other than increasing bucket count) might be
            # helpful, but probably dependent on use-case.
            #
            dist_total: float = 0
            local_min: float = self.rssi_distance_raw or DISTANCE_INFINITE
            for distance in self.hist_distance_by_interval:
                if distance is not None and distance <= local_min:
                    local_min = distance
                dist_total += local_min

            if (_hist_dist_len := len(self.hist_distance_by_interval)) > 0:
                movavg = dist_total / _hist_dist_len
            else:
                movavg = local_min

            # Finally, set the new, smoothed rssi_distance value.
            # The average is only helpful if it's lower than the actual reading.
            if self.rssi_distance_raw is None or movavg < self.rssi_distance_raw:
                self.rssi_distance = movavg
            else:
                self.rssi_distance = self.rssi_distance_raw

        # Deques with maxlen auto-trim, no manual trimming needed for:
        # hist_distance, hist_interval, hist_rssi, hist_stamp, hist_velocity, hist_distance_by_interval

    def to_dict(self):
        """Convert class to serialisable dict for dump_devices."""
        # using "is" comparisons instead of string matching means
        # linting and typing can catch errors.
        out = {}
        for var, val in vars(self).items():
            if val in [self.options]:
                # skip certain vars that we don't want in the dump output.
                continue
            if val in [self.options, self._device, self.scanner_device]:
                # objects we might want to represent but not fully iterate etc.
                out[var] = val.__repr__()
                continue
            if val is self.local_name:
                out[var] = {}
                for namestr, namebytes in self.local_name:
                    out[var][namestr] = namebytes.hex()
                continue
            if val is self.manufacturer_data:
                out[var] = {}
                for manrow in self.manufacturer_data:
                    for manid, manbytes in manrow.items():
                        out[var][manid] = manbytes.hex()
                continue
            if val is self.service_data:
                out[var] = {}
                for svrow in self.service_data:
                    for svid, svbytes in svrow.items():
                        out[var][svid] = svbytes.hex()
                continue
            if isinstance(val, str | int):
                out[var] = val
                continue
            if isinstance(val, float):
                out[var] = round(val, 4)
                continue
            if isinstance(val, list):
                out[var] = []
                for row in val:
                    if isinstance(row, float):
                        out[var].append(round(row, 4))
                    else:
                        out[var].append(row)
                continue
            out[var] = val.__repr__()
        return out

    def __repr__(self) -> str:
        """Help debugging by giving it a clear name instead of empty dict."""
        return f"{self.device_address}__{self.scanner_device.name}"
