"""General helper utilities for Bermuda."""

from __future__ import annotations

import statistics
from functools import lru_cache


@lru_cache(64)
def mac_math_offset(mac, offset=0) -> str | None:
    """
    Perform addition/subtraction on a MAC address.

    With a MAC address in xx:xx:xx:xx:xx:xx format,
    add the offset (which may be negative) to the
    last octet, and return the full new MAC.
    If the resulting octet is outside of 00-FF then
    the function returns None.
    """
    if mac is None:
        return None
    octet = mac[-2:]
    try:
        octet_int = bytes.fromhex(octet)[0]
    except ValueError:
        return None
    if 0 <= (octet_new := octet_int + offset) <= 255:
        return f"{mac[:-3]}:{(octet_new):02x}"
    return None


@lru_cache(1024)
def mac_norm(mac: str) -> str:
    """
    Format the mac address string for entry into dev reg.

    What is returned is always lowercased, regardless of
    detected form.
    If mac is an identifiable MAC-address, it's returned
    in the xx:xx:xx:xx:xx:xx form.

    This is copied from the HA device_registry's
    format_mac, but with a bigger lru cache and some
    tweaks, since we're often dealing with many addresses.
    """
    to_test = mac

    if len(to_test) == 17:
        if to_test.count(":") == 5:
            return to_test.lower()
        if to_test.count("-") == 5:
            return to_test.replace("-", ":").lower()
        if to_test.count("_") == 5:
            return to_test.replace("_", ":").lower()

    elif len(to_test) == 14 and to_test.count(".") == 2:
        to_test = to_test.replace(".", "")

    if len(to_test) == 12:
        # no : included
        return ":".join(to_test.lower()[i : i + 2] for i in range(0, 12, 2))

    # Not sure how formatted, return original
    return mac.lower()


@lru_cache(2048)
def mac_explode_formats(mac: str) -> set[str]:
    """
    Take a formatted mac address and return the formats
    likely to be found in our device info, adverts etc
    by replacing ":" with each of "", "-", "_", ".".
    """
    altmacs = set()
    altmacs.add(mac)
    for newsep in ["", "-", "_", "."]:
        altmacs.add(mac.replace(":", newsep))
    return altmacs


def mac_redact(mac: str, tag: str | None = None) -> str:
    """Remove the centre octets of a MAC and optionally replace with a tag."""
    if tag is None:
        tag = ":"
    return f"{mac[:2]}::{tag}::{mac[-2:]}"


@lru_cache(1024)
def rssi_to_metres(rssi, ref_power=None, attenuation=None):
    """
    Convert instant rssi value to a distance in metres.

    Based on the information from
    https://mdpi-res.com/d_attachment/applsci/applsci-10-02003/article_deploy/applsci-10-02003.pdf?version=1584265508

    Formula: distance = 10^((ref_power - rssi) / (10 * attenuation))

    CRITICAL: Both parameters require per-deployment calibration for accuracy.
    Miscalibration causes exponential distance errors (e.g., 10 dB error = 2-3x distance error).

    attenuation:    Path loss exponent representing environmental signal attenuation.
                    Typical values: 2.5-2.7 (open space), 3.0-3.5 (through walls/obstacles).
                    Will vary by humidity, terrain, building materials, etc.

    ref_power:      Expected RSSI (dBm) when device is at exactly 1 meter from receiver.
                    Typical values: -59 to -75 dBm depending on transmitter power and antenna.
                    MUST be measured for each device type - place beacon at 1m and record RSSI.
                    Affected by receiver sensitivity, transmitter calibration, antenna design, etc.
    """
    if ref_power is None:
        return False
        # ref_power = self.ref_power
    if attenuation is None:
        return False
        # attenuation= self.attenuation

    return 10 ** ((ref_power - rssi) / (10 * attenuation))


def validate_scanners_for_trilateration(
    device,
    current_time: float,
    timeout: float,
    max_scanners: int | None = None,
    debug_enabled: bool = False,
) -> list[tuple]:
    """
    Validate scanner adverts for a device and return valid scanner data.

    Checks each advertisement for:
    - Scanner exists and is marked as scanner
    - Scanner has a configured position
    - Distance is valid (not None and > 0)
    - Distance is within max_radius (configured max distance)
    - Advertisement is not stale (within timeout)

    Results are sorted by distance (closest first), then limited to max_scanners.

    Args:
        device: BermudaDevice to validate scanners for
        current_time: Current timestamp for staleness check
        timeout: Maximum age (seconds) for valid advertisements
        max_scanners: Maximum number of scanners to return (closest are kept)

    Returns:
        List of tuples: [(scanner, advert), ...] for valid scanners, sorted by distance
    """
    from .const import _LOGGER
    from .const import CONF_MAX_RADIUS
    from .const import DEFAULT_MAX_RADIUS

    # Get max_radius configuration
    max_radius = device.options.get(CONF_MAX_RADIUS, DEFAULT_MAX_RADIUS)

    valid_scanners = []

    for advert in device.adverts.values():
        scanner = device._coordinator.devices.get(advert.scanner_address)

        # Scanner must exist and be marked as scanner
        if scanner is None or not scanner.is_scanner:
            if debug_enabled:
                _LOGGER.debug("  - %s: Not a scanner, skipping", advert.scanner_address)
            continue

        # Scanner must have position configured
        if scanner.position is None:
            if debug_enabled:
                _LOGGER.debug("  - %s (%s): No position configured", scanner.name, advert.scanner_address)
            continue

        # Distance must be valid
        if advert.rssi_distance is None or advert.rssi_distance <= 0:
            if debug_enabled:
                _LOGGER.debug("  - %s: Invalid distance (%s)", scanner.name, advert.rssi_distance)
            continue

        # Distance must be within max_radius
        if advert.rssi_distance > max_radius:
            if debug_enabled:
                _LOGGER.debug(
                    "  - %s: Distance %.2fm exceeds max_radius %.2fm",
                    scanner.name,
                    advert.rssi_distance,
                    max_radius,
                )
            continue

        # Advertisement must be recent (not stale)
        advert_age = current_time - advert.stamp
        if advert_age > timeout:
            if debug_enabled:
                _LOGGER.debug(
                    "  - %s: Stale advert (%.1fs old, max %.1fs)",
                    scanner.name,
                    advert_age,
                    timeout,
                )
            continue

        # This scanner is valid!
        valid_scanners.append((scanner, advert))
        if debug_enabled:
            _LOGGER.debug(
                "  ✓ %s [%s]: pos=%s dist=%.2fm age=%.1fs",
                scanner.name,
                scanner.address,
                scanner.position,
                advert.rssi_distance,
                advert_age,
            )

    # Sort by distance (closest first), then by recency as tiebreaker
    valid_scanners.sort(key=lambda x: (x[1].rssi_distance, -(x[1].stamp)))

    # Detect large distance gaps that might indicate floor separation
    # If there's a gap >3m between consecutive scanners, exclude the distant ones
    if len(valid_scanners) > 3:
        distances = [advert.rssi_distance for _, advert in valid_scanners]
        for i in range(1, len(distances)):
            gap = distances[i] - distances[i - 1]
            if gap > 3.0 and i >= 3:  # Large gap after at least 3 close scanners
                if debug_enabled:
                    _LOGGER.debug(
                        "Detected distance gap of %.1fm between scanner %d (%.1fm) and %d (%.1fm) - "
                        "likely floor separation, excluding %d distant scanners",
                        gap,
                        i - 1,
                        distances[i - 1],
                        i,
                        distances[i],
                        len(valid_scanners) - i,
                    )
                valid_scanners = valid_scanners[:i]
                break
    
    # Floor clustering and z-height filtering
    # Instead of using single closest scanner, use median z of closest scanners for more robust filtering
    if len(valid_scanners) >= 3:
        # Get z-coordinates of scanners
        z_coords = [scanner.position[2] for scanner, _ in valid_scanners]
        
        # Use median z of 3-5 closest scanners (more robust than single closest)
        num_for_median = min(5, len(valid_scanners))
        closest_scanners_z = [valid_scanners[i][0].position[2] for i in range(num_for_median)]
        reference_z = statistics.median(closest_scanners_z)
        
        # Check if there's significant z-variation (>1.5m, roughly one floor)
        z_range = max(z_coords) - min(z_coords)
        if z_range > 1.5:
            # Floor clustering: group scanners by floor and select best cluster
            # Group scanners within 1.5m z-height (same floor)
            floor_groups = {}
            for scanner, advert in valid_scanners:
                scanner_z = scanner.position[2]
                # Find existing floor group within 1.5m
                found_group = False
                for group_z in floor_groups:
                    if abs(scanner_z - group_z) <= 1.5:
                        floor_groups[group_z].append((scanner, advert))
                        found_group = True
                        break
                if not found_group:
                    # Create new floor group
                    floor_groups[scanner_z] = [(scanner, advert)]
            
            # Select floor with most scanners AND closest to reference_z
            # Prefer floor with better scanner coverage
            best_floor_z = reference_z
            best_floor_score = -1
            for group_z, scanners in floor_groups.items():
                # Score: number of scanners + bonus if close to reference_z
                z_bonus = 3 if abs(group_z - reference_z) <= 1.5 else 0
                score = len(scanners) + z_bonus
                if score > best_floor_score:
                    best_floor_score = score
                    best_floor_z = group_z
            
            # Filter to scanners on best floor
            filtered_scanners = [
                (scanner, advert) for scanner, advert in valid_scanners
                if abs(scanner.position[2] - best_floor_z) <= 1.5
            ]
            
            if len(filtered_scanners) >= 3:
                if debug_enabled:
                    _LOGGER.debug(
                        "Floor clustering: %d floors detected, selected floor at z=%.1fm with %d scanners "
                        "(reference_z=%.1fm from %d closest, z_range=%.1fm)",
                        len(floor_groups),
                        best_floor_z,
                        len(filtered_scanners),
                        reference_z,
                        num_for_median,
                        z_range,
                    )
                valid_scanners = filtered_scanners
            else:
                if debug_enabled:
                    _LOGGER.debug(
                        "Floor clustering would reduce scanners to %d (< 3), keeping all %d scanners",
                        len(filtered_scanners),
                        len(valid_scanners),
                    )
        else:
            if debug_enabled:
                _LOGGER.debug(
                    "All scanners on same floor: z_range=%.1fm <= 1.5m, reference_z=%.1fm",
                    z_range,
                    reference_z,
                )

    # Limit to max_scanners if specified
    if max_scanners is not None and len(valid_scanners) > max_scanners:
        if debug_enabled:
            _LOGGER.debug(
                "Limiting scanners from %d to %d closest for trilateration",
                len(valid_scanners),
                max_scanners,
            )
        valid_scanners = valid_scanners[:max_scanners]

    # Log summary of all valid scanners for debugging (always logged, not just in debug mode)
    if valid_scanners:
        scanner_summary = []
        for scanner, advert in valid_scanners:
            scanner_summary.append(
                f"{scanner.name}[{scanner.address}] pos={scanner.position} dist={advert.rssi_distance:.2f}m"
            )
        _LOGGER.debug(
            "Valid scanners for %s (%d): %s",
            device.name,
            len(valid_scanners),
            ", ".join(scanner_summary),
        )

    return valid_scanners


@lru_cache(256)
def clean_charbuf(instring: str | None) -> str:
    """
    Some people writing C on bluetooth devices seem to
    get confused between char arrays, strings and such. This
    function takes a potentially dodgy charbuf from a bluetooth
    device and cleans it of leading/trailing cruft
    and returns what's left, up to the first null, if any.

    If given None it returns an empty string.
    Characters trimmed are space, tab, CR, LF, NUL.
    """
    if instring is not None:
        return instring.strip(" \t\r\n\x00").split("\0")[0]
    return ""


def calculate_distance_from_position(
    scanner_pos: tuple[float, float, float],
    target_pos: tuple[float, float, float],
) -> float:
    """
    Calculate 3D Euclidean distance between scanner and target position.

    Args:
        scanner_pos: (x, y, z) coordinates of scanner in meters
        target_pos: (x, y, z) coordinates of target in meters

    Returns:
        Distance in meters
    """
    return (
        (scanner_pos[0] - target_pos[0]) ** 2
        + (scanner_pos[1] - target_pos[1]) ** 2
        + (scanner_pos[2] - target_pos[2]) ** 2
    ) ** 0.5


def calculate_scanner_rssi_offsets(
    scanner_rssi_data: dict[str, list[float]],
    scanner_positions: dict[str, tuple[float, float, float]],
    target_position: tuple[float, float, float],
    ref_power: float,
    attenuation: float,
    mode: str = "offsets_only",
) -> dict[str, float]:
    """
    Calculate per-scanner RSSI offsets using a calibration beacon at known position.

    This implements MVP auto-calibration: given a beacon at a known location and
    multiple scanners with known positions, compute RSSI offsets that normalize
    scanner sensitivity differences.

    Algorithm:
    1. For each scanner, compute actual distance from scanner to beacon using positions
    2. Compute expected RSSI using the log-distance path-loss model with current ref_power/attenuation
    3. Compute median observed RSSI from samples
    4. RSSI offset = expected_rssi - observed_rssi (additive correction)

    Args:
        scanner_rssi_data: Dict mapping scanner_address -> list of RSSI samples (floats)
        scanner_positions: Dict mapping scanner_address -> (x, y, z) position in meters
        target_position: (x, y, z) position of calibration beacon in meters
        ref_power: Current global ref_power parameter (dBm)
        attenuation: Current global attenuation parameter (path loss exponent)
        mode: "offsets_only" (default) or "full" (future: also fit ref_power/attenuation)

    Returns:
        Dict mapping scanner_address -> rssi_offset (dBm, additive)
        Empty dict if insufficient data

    Raises:
        ValueError: If mode is unsupported or inputs are invalid
    """
    import math

    if mode not in ("offsets_only", "full"):
        raise ValueError(f"Unsupported calibration mode: {mode}")

    if mode == "full":
        raise NotImplementedError("Full calibration mode (fitting ref_power/attenuation) not yet implemented")

    offsets = {}

    # Validate we have at least 2 scanners with data and positions
    common_scanners = set(scanner_rssi_data.keys()) & set(scanner_positions.keys())
    if len(common_scanners) < 2:
        return {}

    for scanner_addr in common_scanners:
        rssi_samples = scanner_rssi_data[scanner_addr]
        scanner_pos = scanner_positions[scanner_addr]

        # Need enough samples for robust median
        if len(rssi_samples) < 10:
            continue

        # Calculate actual distance from scanner to beacon
        distance = calculate_distance_from_position(scanner_pos, target_position)

        # Prevent division issues with very small distances
        if distance < 0.1:
            continue

        # Calculate expected RSSI at this distance using current model
        # Formula: rssi_expected = ref_power - 10 * attenuation * log10(distance)
        expected_rssi = ref_power - 10 * attenuation * math.log10(distance)

        # Get robust observed RSSI (median filters outliers)
        observed_rssi = statistics.median(rssi_samples)

        # Offset is the correction needed: offset + observed = expected
        # So offset = expected - observed
        offset = expected_rssi - observed_rssi

        offsets[scanner_addr] = offset

    return offsets


def get_calibration_quality_metrics(
    scanner_rssi_data: dict[str, list[float]],
    scanner_positions: dict[str, tuple[float, float, float]],
    target_position: tuple[float, float, float],
) -> dict[str, dict[str, float]]:
    """
    Calculate quality metrics for calibration data to help user assess reliability.

    Returns per-scanner metrics including:
    - sample_count: Number of RSSI samples
    - rssi_median: Median RSSI value (dBm)
    - rssi_std: Standard deviation of RSSI (dBm, indicates stability)
    - distance: Physical distance from scanner to beacon (meters)

    Args:
        scanner_rssi_data: Dict mapping scanner_address -> list of RSSI samples
        scanner_positions: Dict mapping scanner_address -> (x, y, z) position
        target_position: (x, y, z) position of calibration beacon

    Returns:
        Dict mapping scanner_address -> metrics dict
    """
    metrics = {}

    common_scanners = set(scanner_rssi_data.keys()) & set(scanner_positions.keys())

    for scanner_addr in common_scanners:
        rssi_samples = scanner_rssi_data[scanner_addr]
        scanner_pos = scanner_positions[scanner_addr]

        if len(rssi_samples) < 2:
            continue

        distance = calculate_distance_from_position(scanner_pos, target_position)

        metrics[scanner_addr] = {
            "sample_count": len(rssi_samples),
            "rssi_median": statistics.median(rssi_samples),
            "rssi_std": statistics.stdev(rssi_samples) if len(rssi_samples) > 1 else 0.0,
            "distance": distance,
        }

    return metrics


def calculate_scanner_offsets_from_scanner_pairs(
    scanner_pairs_rssi: dict[tuple[str, str], list[float]],
    scanner_positions: dict[str, tuple[float, float, float]],
    ref_power: float,
    attenuation: float,
) -> dict[str, float]:
    """
    Calculate per-scanner RSSI offsets using scanner-to-scanner RF ranging.

    This algorithm uses scanners measuring each other instead of an external beacon.
    For each scanner pair (A, B) where scanner A receives from scanner B:
    1. Calculate actual distance between scanners using their positions
    2. Calculate expected RSSI using log-distance model
    3. Get median observed RSSI
    4. Solve for per-scanner offsets that minimize total error

    The offset model:
    - Each scanner has a TX offset (when transmitting) and RX offset (when receiving)
    - For simplicity, we assume TX_offset = RX_offset for each scanner
    - observed_RSSI(A receives from B) = expected_RSSI - offset_B + offset_A
      (B's transmission is affected by its offset, A's reception adds its offset)

    Algorithm:
    - Set one scanner as reference (offset = 0)
    - For each other scanner, calculate offset from all measurements involving it
    - Use weighted average across multiple pair measurements

    Args:
        scanner_pairs_rssi: Dict mapping (receiver_addr, transmitter_addr) -> RSSI samples
        scanner_positions: Dict mapping scanner_addr -> (x, y, z) position
        ref_power: Global ref_power parameter (dBm)
        attenuation: Global attenuation parameter (path loss exponent)

    Returns:
        Dict mapping scanner_address -> rssi_offset (dBm, additive)
        Empty dict if insufficient data

    Raises:
        ValueError: If inputs are invalid
    """
    import math

    if not scanner_pairs_rssi:
        return {}

    # Get all scanners involved in measurements
    all_scanners = set()
    for (receiver, transmitter) in scanner_pairs_rssi.keys():
        all_scanners.add(receiver)
        all_scanners.add(transmitter)

    # Filter to only scanners with known positions
    scanners_with_positions = {s for s in all_scanners if s in scanner_positions}

    if len(scanners_with_positions) < 2:
        return {}

    # Choose reference scanner (most measurements, or alphabetically first)
    reference_scanner = min(scanners_with_positions)

    # Calculate offsets for each scanner relative to reference
    scanner_offsets = {reference_scanner: 0.0}

    # For each scanner, collect all measurements involving it
    for scanner_addr in scanners_with_positions:
        if scanner_addr == reference_scanner:
            continue

        offset_estimates = []

        # Case 1: This scanner received from reference scanner
        pair_key_rx = (scanner_addr, reference_scanner)
        if pair_key_rx in scanner_pairs_rssi:
            rssi_samples = scanner_pairs_rssi[pair_key_rx]
            if len(rssi_samples) >= 5:
                # Calculate expected distance
                pos1 = scanner_positions[scanner_addr]
                pos2 = scanner_positions[reference_scanner]
                distance = calculate_distance_from_position(pos1, pos2)

                if distance >= 0.1:  # Avoid division issues
                    # Expected RSSI at this distance
                    expected_rssi = ref_power - 10 * attenuation * math.log10(distance)

                    # Observed RSSI (median)
                    observed_rssi = statistics.median(rssi_samples)

                    # observed = expected - offset_transmitter + offset_receiver
                    # observed = expected - offset_ref + offset_scanner
                    # observed = expected - 0 + offset_scanner
                    # offset_scanner = observed - expected
                    offset = observed_rssi - expected_rssi
                    offset_estimates.append(offset)

        # Case 2: Reference scanner received from this scanner
        pair_key_tx = (reference_scanner, scanner_addr)
        if pair_key_tx in scanner_pairs_rssi:
            rssi_samples = scanner_pairs_rssi[pair_key_tx]
            if len(rssi_samples) >= 5:
                pos1 = scanner_positions[reference_scanner]
                pos2 = scanner_positions[scanner_addr]
                distance = calculate_distance_from_position(pos1, pos2)

                if distance >= 0.1:
                    expected_rssi = ref_power - 10 * attenuation * math.log10(distance)
                    observed_rssi = statistics.median(rssi_samples)

                    # observed = expected - offset_transmitter + offset_receiver
                    # observed = expected - offset_scanner + offset_ref
                    # observed = expected - offset_scanner + 0
                    # offset_scanner = expected - observed
                    offset = expected_rssi - observed_rssi
                    offset_estimates.append(offset)

        # Case 3: This scanner and another non-reference scanner
        for other_scanner in scanners_with_positions:
            if other_scanner in (scanner_addr, reference_scanner):
                continue

            # This scanner received from other
            pair_key = (scanner_addr, other_scanner)
            if pair_key in scanner_pairs_rssi:
                rssi_samples = scanner_pairs_rssi[pair_key]
                if len(rssi_samples) >= 5 and other_scanner in scanner_offsets:
                    pos1 = scanner_positions[scanner_addr]
                    pos2 = scanner_positions[other_scanner]
                    distance = calculate_distance_from_position(pos1, pos2)

                    if distance >= 0.1:
                        expected_rssi = ref_power - 10 * attenuation * math.log10(distance)
                        observed_rssi = statistics.median(rssi_samples)

                        # observed = expected - offset_transmitter + offset_receiver
                        # observed = expected - offset_other + offset_scanner
                        # offset_scanner = observed - expected + offset_other
                        other_offset = scanner_offsets[other_scanner]
                        offset = observed_rssi - expected_rssi + other_offset
                        offset_estimates.append(offset)

            # Other scanner received from this one
            pair_key = (other_scanner, scanner_addr)
            if pair_key in scanner_pairs_rssi:
                rssi_samples = scanner_pairs_rssi[pair_key]
                if len(rssi_samples) >= 5 and other_scanner in scanner_offsets:
                    pos1 = scanner_positions[other_scanner]
                    pos2 = scanner_positions[scanner_addr]
                    distance = calculate_distance_from_position(pos1, pos2)

                    if distance >= 0.1:
                        expected_rssi = ref_power - 10 * attenuation * math.log10(distance)
                        observed_rssi = statistics.median(rssi_samples)

                        # observed = expected - offset_transmitter + offset_receiver
                        # observed = expected - offset_scanner + offset_other
                        # offset_scanner = expected - observed + offset_other
                        other_offset = scanner_offsets[other_scanner]
                        offset = expected_rssi - observed_rssi + other_offset
                        offset_estimates.append(offset)

        # Calculate average offset from all estimates
        if offset_estimates:
            scanner_offsets[scanner_addr] = statistics.mean(offset_estimates)

    # Only return if we have at least 2 scanners calibrated
    if len(scanner_offsets) < 2:
        return {}

    return scanner_offsets


def get_scanner_pair_quality_metrics(
    scanner_pairs_rssi: dict[tuple[str, str], list[float]],
    scanner_positions: dict[str, tuple[float, float, float]],
) -> dict[tuple[str, str], dict[str, float]]:
    """
    Calculate quality metrics for scanner-to-scanner calibration data.

    Returns per-pair metrics to help user assess data reliability.

    Args:
        scanner_pairs_rssi: Dict mapping (receiver, transmitter) -> RSSI samples
        scanner_positions: Dict mapping scanner_address -> (x, y, z) position

    Returns:
        Dict mapping (receiver, transmitter) -> metrics dict with:
        - sample_count: Number of RSSI samples
        - rssi_median: Median RSSI value (dBm)
        - rssi_std: Standard deviation of RSSI (dBm)
        - distance: Physical distance between scanners (meters)
        - receiver_name: Name of receiving scanner
        - transmitter_name: Name of transmitting scanner
    """
    metrics = {}

    for (receiver_addr, transmitter_addr), rssi_samples in scanner_pairs_rssi.items():
        if len(rssi_samples) < 2:
            continue

        if receiver_addr not in scanner_positions or transmitter_addr not in scanner_positions:
            continue

        receiver_pos = scanner_positions[receiver_addr]
        transmitter_pos = scanner_positions[transmitter_addr]
        distance = calculate_distance_from_position(receiver_pos, transmitter_pos)

        metrics[(receiver_addr, transmitter_addr)] = {
            "sample_count": len(rssi_samples),
            "rssi_median": statistics.median(rssi_samples),
            "rssi_std": statistics.stdev(rssi_samples) if len(rssi_samples) > 1 else 0.0,
            "distance": distance,
        }

    return metrics
