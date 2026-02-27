"""Auto-optimization for Bermuda BLE parameters using Nelder-Mead.

Implements ESPresense-style parameter optimization using scanner-to-scanner
RF ranging data. Three independent optimizers:
1. Per-scanner RSSI offsets (rx_adj)
2. Global absorption/attenuation
3. Global ref_power (tx_ref_rssi)

Each uses Nelder-Mead simplex minimization of RMS distance error,
with bounded parameters and change gating (only apply if error improves).
"""

from __future__ import annotations

import math
import statistics
from typing import TYPE_CHECKING

from .const import _LOGGER

if TYPE_CHECKING:
    pass

# ESPresense-compatible parameter bounds
ABSORPTION_MIN = 2.5
ABSORPTION_MAX = 3.5
REF_POWER_MIN = -70.0
REF_POWER_MAX = -50.0
RX_ADJ_MIN = -15.0
RX_ADJ_MAX = 20.0
BASELINE_SNAPSHOTS = 3


def calculate_rms_error(
    scanner_pairs_rssi: dict[tuple[str, str], list[float]],
    scanner_positions: dict[str, tuple[float, float, float]],
    ref_power: float,
    attenuation: float,
    offsets: dict[str, float] | None = None,
) -> float:
    """Calculate total RMS distance error across all scanner pairs.

    For each pair (receiver, transmitter):
    - True distance from configured positions (Euclidean)
    - Predicted distance from RSSI using log-distance model with offsets
    - Error = (predicted - true)²

    Returns sqrt(mean(errors)), or inf if no valid pairs.
    """
    if offsets is None:
        offsets = {}

    errors_sq = []
    for (receiver, transmitter), rssi_samples in scanner_pairs_rssi.items():
        if receiver not in scanner_positions or transmitter not in scanner_positions:
            continue
        if len(rssi_samples) < 3:
            continue

        # True distance from configured positions
        pos_r = scanner_positions[receiver]
        pos_t = scanner_positions[transmitter]
        true_dist = math.sqrt(
            (pos_r[0] - pos_t[0]) ** 2
            + (pos_r[1] - pos_t[1]) ** 2
            + (pos_r[2] - pos_t[2]) ** 2
        )

        if true_dist < 0.1:
            continue

        # Median observed RSSI
        observed_rssi = statistics.median(rssi_samples)

        # Apply offsets: observed = true_rssi + offset_rx - offset_tx
        # So adjusted = observed - offset_rx + offset_tx
        adjusted_rssi = observed_rssi - offsets.get(receiver, 0.0) + offsets.get(transmitter, 0.0)

        # Predicted distance from log-distance model
        if attenuation <= 0:
            continue
        predicted_dist = 10 ** ((ref_power - adjusted_rssi) / (10 * attenuation))

        errors_sq.append((predicted_dist - true_dist) ** 2)

    if not errors_sq:
        return float("inf")

    return math.sqrt(sum(errors_sq) / len(errors_sq))


def _nelder_mead_1d(
    objective,
    initial: float,
    lower: float,
    upper: float,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> tuple[float, float]:
    """Nelder-Mead-style golden section search for 1D bounded optimization.

    Returns (best_param, best_value).
    """
    # Golden section search is more efficient for 1D than full simplex
    gr = (math.sqrt(5) + 1) / 2  # golden ratio

    a, b = lower, upper

    # Narrow with golden section
    c = b - (b - a) / gr
    d = a + (b - a) / gr

    for _ in range(max_iterations):
        if abs(b - a) < tolerance:
            break

        fc = objective(c)
        fd = objective(d)

        if fc < fd:
            b = d
        else:
            a = c

        c = b - (b - a) / gr
        d = a + (b - a) / gr

    best = (a + b) / 2.0
    return best, objective(best)


def _nelder_mead_nd(
    objective,
    initial: list[float],
    lower: list[float],
    upper: list[float],
    max_iterations: int = 200,
    tolerance: float = 1e-4,
) -> tuple[list[float], float]:
    """Nelder-Mead simplex optimization for N-dimensional bounded problems.

    Returns (best_params, best_value).
    """
    n = len(initial)

    def clamp(point: list[float]) -> list[float]:
        return [max(lower[i], min(upper[i], point[i])) for i in range(n)]

    # Create initial simplex
    step = 0.5
    simplex = [initial[:]]
    for d in range(n):
        point = initial[:]
        # Step toward center of bounds if near edge
        if initial[d] + step > upper[d]:
            point[d] = initial[d] - step
        else:
            point[d] = initial[d] + step
        simplex.append(clamp(point))

    values = [objective(s) for s in simplex]

    # Standard Nelder-Mead coefficients
    alpha = 1.0   # reflection
    gamma = 2.0   # expansion
    rho = 0.5     # contraction
    sigma = 0.5   # shrink

    for _ in range(max_iterations):
        # Sort simplex by function value
        order = sorted(range(len(values)), key=lambda i: values[i])
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]

        # Check convergence
        if values[-1] - values[0] < tolerance:
            break

        # Centroid of all points except worst
        c = [0.0] * n
        for i in range(len(simplex) - 1):
            for d in range(n):
                c[d] += simplex[i][d]
        for d in range(n):
            c[d] /= (len(simplex) - 1)

        worst = simplex[-1]

        # Reflection
        xr = clamp([c[d] + alpha * (c[d] - worst[d]) for d in range(n)])
        fr = objective(xr)

        if values[0] <= fr < values[-2]:
            simplex[-1] = xr
            values[-1] = fr
        elif fr < values[0]:
            # Expansion
            xe = clamp([c[d] + gamma * (xr[d] - c[d]) for d in range(n)])
            fe = objective(xe)
            if fe < fr:
                simplex[-1] = xe
                values[-1] = fe
            else:
                simplex[-1] = xr
                values[-1] = fr
        else:
            # Contraction
            xc = clamp([c[d] + rho * (worst[d] - c[d]) for d in range(n)])
            fc = objective(xc)
            if fc < values[-1]:
                simplex[-1] = xc
                values[-1] = fc
            else:
                # Shrink
                best = simplex[0]
                for i in range(1, len(simplex)):
                    simplex[i] = clamp(
                        [best[d] + sigma * (simplex[i][d] - best[d]) for d in range(n)]
                    )
                    values[i] = objective(simplex[i])

    # Return best
    best_idx = min(range(len(values)), key=lambda i: values[i])
    return simplex[best_idx], values[best_idx]


def optimize_absorption(
    scanner_pairs_rssi: dict[tuple[str, str], list[float]],
    scanner_positions: dict[str, tuple[float, float, float]],
    ref_power: float,
    current_attenuation: float,
    offsets: dict[str, float] | None = None,
) -> tuple[float, float]:
    """Optimize global attenuation using Nelder-Mead.

    Returns (best_attenuation, rms_error).
    Bounds: 2.5 to 3.5 (ESPresense defaults).
    """
    def objective(att: float) -> float:
        return calculate_rms_error(
            scanner_pairs_rssi, scanner_positions, ref_power, att, offsets
        )

    best_att, best_err = _nelder_mead_1d(
        objective,
        initial=current_attenuation,
        lower=ABSORPTION_MIN,
        upper=ABSORPTION_MAX,
    )
    return best_att, best_err


def optimize_ref_power(
    scanner_pairs_rssi: dict[tuple[str, str], list[float]],
    scanner_positions: dict[str, tuple[float, float, float]],
    current_ref_power: float,
    attenuation: float,
    offsets: dict[str, float] | None = None,
) -> tuple[float, float]:
    """Optimize global ref_power using Nelder-Mead.

    Returns (best_ref_power, rms_error).
    Bounds: -70 to -50 (ESPresense defaults).
    """
    def objective(rp: float) -> float:
        return calculate_rms_error(
            scanner_pairs_rssi, scanner_positions, rp, attenuation, offsets
        )

    best_rp, best_err = _nelder_mead_1d(
        objective,
        initial=current_ref_power,
        lower=REF_POWER_MIN,
        upper=REF_POWER_MAX,
    )
    return best_rp, best_err


def optimize_offsets(
    scanner_pairs_rssi: dict[tuple[str, str], list[float]],
    scanner_positions: dict[str, tuple[float, float, float]],
    ref_power: float,
    attenuation: float,
) -> tuple[dict[str, float], float]:
    """Optimize per-scanner RSSI offsets jointly using Nelder-Mead.

    Returns (offsets_dict, rms_error).
    Bounds: -15 to +20 per scanner (ESPresense defaults).
    First scanner is held at 0 as reference.
    """
    # Get all scanners involved
    all_scanners = set()
    for receiver, transmitter in scanner_pairs_rssi:
        if receiver in scanner_positions:
            all_scanners.add(receiver)
        if transmitter in scanner_positions:
            all_scanners.add(transmitter)

    scanner_list = sorted(all_scanners)
    if len(scanner_list) < 2:
        return {}, float("inf")

    # First scanner is reference (offset = 0)
    reference = scanner_list[0]
    optimizable = scanner_list[1:]
    n = len(optimizable)

    def objective(params: list[float]) -> float:
        offsets = {reference: 0.0}
        for i, addr in enumerate(optimizable):
            offsets[addr] = params[i]
        return calculate_rms_error(
            scanner_pairs_rssi, scanner_positions, ref_power, attenuation, offsets
        )

    initial = [0.0] * n
    lower = [RX_ADJ_MIN] * n
    upper = [RX_ADJ_MAX] * n

    best_params, best_err = _nelder_mead_nd(
        objective, initial, lower, upper, max_iterations=300
    )

    offsets = {reference: 0.0}
    for i, addr in enumerate(optimizable):
        offsets[addr] = round(best_params[i], 1)

    return offsets, best_err


class OptimizationBaseline:
    """Tracks baseline RMS error for change gating.

    ESPresense requires 3 snapshots to establish baseline error
    before allowing parameter changes.
    """

    def __init__(self, required_snapshots: int = BASELINE_SNAPSHOTS) -> None:
        self.required_snapshots = required_snapshots
        self._snapshots: list[float] = []
        self._baseline: float | None = None

    @property
    def is_established(self) -> bool:
        """Whether enough snapshots have been collected."""
        return self._baseline is not None

    @property
    def baseline_error(self) -> float | None:
        """The established baseline RMS error."""
        return self._baseline

    def add_snapshot(self, rms_error: float) -> bool:
        """Add an RMS error snapshot. Returns True if baseline is now established."""
        if self._baseline is not None:
            return True

        self._snapshots.append(rms_error)
        _LOGGER.info(
            "Optimization baseline: snapshot %d/%d, RMS error=%.4f",
            len(self._snapshots),
            self.required_snapshots,
            rms_error,
        )

        if len(self._snapshots) >= self.required_snapshots:
            self._baseline = statistics.mean(self._snapshots)
            _LOGGER.info(
                "Optimization baseline established: RMS error=%.4f (from %d snapshots)",
                self._baseline,
                len(self._snapshots),
            )
            return True

        return False

    def should_apply(self, new_rms_error: float) -> bool:
        """Check if new parameters should be applied (error improved vs baseline)."""
        if self._baseline is None:
            return False
        return new_rms_error < self._baseline

    def update_baseline(self, new_rms_error: float) -> None:
        """Update baseline after successful parameter application."""
        self._baseline = new_rms_error
