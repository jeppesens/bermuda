"""General helper utilities for Bermuda."""

from __future__ import annotations

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

    attenuation:    a factor representing environmental attenuation
                    along the path. Will vary by humidity, terrain etc.
    ref_power:      db. measured rssi when at 1m distance from rx. The will
                    be affected by both receiver sensitivity and transmitter
                    calibration, antenna design and orientation etc.
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
) -> list[tuple]:
    """
    Validate scanner adverts for a device and return valid scanner data.

    Checks each advertisement for:
    - Scanner exists and is marked as scanner
    - Scanner has a configured position
    - Distance is valid (not None and > 0)
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

    valid_scanners = []

    for advert in device.adverts.values():
        scanner = device._coordinator.devices.get(advert.scanner_address)

        # Scanner must exist and be marked as scanner
        if scanner is None or not scanner.is_scanner:
            _LOGGER.debug("  - %s: Not a scanner, skipping", advert.scanner_address)
            continue

        # Scanner must have position configured
        if scanner.position is None:
            _LOGGER.debug("  - %s (%s): No position configured", scanner.name, advert.scanner_address)
            continue

        # Distance must be valid
        if advert.rssi_distance is None or advert.rssi_distance <= 0:
            _LOGGER.debug("  - %s: Invalid distance (%s)", scanner.name, advert.rssi_distance)
            continue

        # Advertisement must be recent (not stale)
        advert_age = current_time - advert.stamp
        if advert_age > timeout:
            _LOGGER.debug(
                "  - %s: Stale advert (%.1fs old, max %.1fs)",
                scanner.name,
                advert_age,
                timeout,
            )
            continue

        # This scanner is valid!
        valid_scanners.append((scanner, advert))
        _LOGGER.debug(
            "  ✓ %s: pos=%s dist=%.2fm age=%.1fs",
            scanner.name,
            scanner.position,
            advert.rssi_distance,
            advert_age,
        )

    # Sort by distance (closest first), then by recency (newer first) as tiebreaker
    valid_scanners.sort(key=lambda x: (x[1].rssi_distance, -(x[1].stamp)))

    # Detect large distance gaps that might indicate floor separation
    # If there's a gap >3m between consecutive scanners, exclude the distant ones
    if len(valid_scanners) > 3:
        distances = [advert.rssi_distance for _, advert in valid_scanners]
        for i in range(1, len(distances)):
            gap = distances[i] - distances[i - 1]
            if gap > 3.0 and i >= 3:  # Large gap after at least 3 close scanners
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
    
    # Filter by z-height if we detect scanners at significantly different heights
    # This helps when device is upstairs but some downstairs scanners are also hearing it
    if len(valid_scanners) >= 3:
        # Get z-coordinates of scanners
        z_coords = [scanner.position[2] for scanner, _ in valid_scanners]
        closest_z = valid_scanners[0][0].position[2]  # z of closest scanner
        
        # Check if there's significant z-variation (>1.5m, roughly one floor)
        z_range = max(z_coords) - min(z_coords)
        if z_range > 1.5:
            # Filter out scanners that are >1.5m different in z from the closest scanner
            # This assumes floors are separated by at least 1.5m
            filtered_scanners = [
                (scanner, advert) for scanner, advert in valid_scanners
                if abs(scanner.position[2] - closest_z) <= 1.5
            ]
            
            if len(filtered_scanners) >= 3:
                _LOGGER.debug(
                    "Filtered scanners by z-height: %d -> %d (closest_z=%.1fm, z_range=%.1fm)",
                    len(valid_scanners),
                    len(filtered_scanners),
                    closest_z,
                    z_range,
                )
                valid_scanners = filtered_scanners

    # Limit to max_scanners if specified
    if max_scanners is not None and len(valid_scanners) > max_scanners:
        _LOGGER.debug(
            "Limiting scanners from %d to %d closest for trilateration",
            len(valid_scanners),
            max_scanners,
        )
        valid_scanners = valid_scanners[:max_scanners]

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
