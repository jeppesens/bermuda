"""
Kalman filter for 3D position tracking with velocity estimation.

Ported from ESPresense-companion's KalmanLocation.cs.
Implements a 6-state Kalman filter tracking [x, y, z, vx, vy, vz]
using a constant-velocity motion model.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass


@dataclass
class KalmanFilterSettings:
    """Settings for the Kalman location filter.

    Matches ESPresense-companion defaults.
    """

    process_noise: float = 0.01
    measurement_noise: float = 0.1
    max_velocity: float = 0.5


# Pre-allocated identity matrix for reuse
_IDENTITY_6 = [
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
]

# Measurement matrix H: maps state [x,y,z,vx,vy,vz] -> measurement [x,y,z]
_H = [
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
]

# H transposed (6x3)
_HT = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
]


def _mat_mul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    """Multiply two matrices."""
    rows_a = len(a)
    cols_b = len(b[0])
    cols_a = len(a[0])
    result = [[0.0] * cols_b for _ in range(rows_a)]
    for i in range(rows_a):
        for j in range(cols_b):
            s = 0.0
            for k in range(cols_a):
                s += a[i][k] * b[k][j]
            result[i][j] = s
    return result


def _mat_transpose(m: list[list[float]]) -> list[list[float]]:
    """Transpose a matrix."""
    rows = len(m)
    cols = len(m[0])
    return [[m[i][j] for i in range(rows)] for j in range(cols)]


def _mat_add(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    """Add two matrices."""
    return [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def _mat_sub(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    """Subtract matrix b from a."""
    return [[a[i][j] - b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def _mat_scale(m: list[list[float]], s: float) -> list[list[float]]:
    """Scale a matrix by a scalar."""
    return [[m[i][j] * s for j in range(len(m[0]))] for i in range(len(m))]


def _inv3x3(m: list[list[float]]) -> list[list[float]]:
    """Invert a 3x3 matrix using Cramer's rule."""
    a, b, c = m[0]
    d, e, f = m[1]
    g, h, i = m[2]

    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if abs(det) < 1e-15:
        # Nearly singular - return large diagonal (effectively ignore measurement)
        return [[1e6, 0.0, 0.0], [0.0, 1e6, 0.0], [0.0, 0.0, 1e6]]

    inv_det = 1.0 / det
    return [
        [
            (e * i - f * h) * inv_det,
            (c * h - b * i) * inv_det,
            (b * f - c * e) * inv_det,
        ],
        [
            (f * g - d * i) * inv_det,
            (a * i - c * g) * inv_det,
            (c * d - a * f) * inv_det,
        ],
        [
            (d * h - e * g) * inv_det,
            (b * g - a * h) * inv_det,
            (a * e - b * d) * inv_det,
        ],
    ]


def _make_state_transition(dt: float) -> list[list[float]]:
    """Create state transition matrix F for constant-velocity model."""
    return [
        [1.0, 0.0, 0.0, dt, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, dt, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, dt],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ]


def _make_process_noise(dt: float, q: float) -> list[list[float]]:
    """Create process noise matrix Q for constant-velocity model."""
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt3 * dt
    return [
        [q * dt4 / 4, 0.0, 0.0, q * dt3 / 2, 0.0, 0.0],
        [0.0, q * dt4 / 4, 0.0, 0.0, q * dt3 / 2, 0.0],
        [0.0, 0.0, q * dt4 / 4, 0.0, 0.0, q * dt3 / 2],
        [q * dt3 / 2, 0.0, 0.0, q * dt2, 0.0, 0.0],
        [0.0, q * dt3 / 2, 0.0, 0.0, q * dt2, 0.0],
        [0.0, 0.0, q * dt3 / 2, 0.0, 0.0, q * dt2],
    ]


class KalmanLocation:
    """Kalman filter for 3D position tracking with velocity estimation.

    Tracks a 6-state vector [x, y, z, vx, vy, vz] using a constant-velocity
    motion model. Ported from ESPresense-companion's KalmanLocation.cs.
    """

    def __init__(self, settings: KalmanFilterSettings | None = None) -> None:
        """Initialize Kalman filter."""
        self.settings = settings or KalmanFilterSettings()
        self._state: list[list[float]] | None = None  # 6x1 column vector
        self._covariance: list[list[float]] | None = None  # 6x6 error covariance
        self._last_update: float | None = None
        self.location: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def velocity(self) -> tuple[float, float, float]:
        """Return estimated velocity (vx, vy, vz)."""
        if self._state is None:
            return (0.0, 0.0, 0.0)
        return (self._state[3][0], self._state[4][0], self._state[5][0])

    @property
    def speed(self) -> float:
        """Return speed magnitude in m/s."""
        vx, vy, vz = self.velocity
        return math.sqrt(vx * vx + vy * vy + vz * vz)

    def reset(self, x: float, y: float, z: float) -> None:
        """Reset filter to a new position with zero velocity."""
        self._state = [[x], [y], [z], [0.0], [0.0], [0.0]]
        self._covariance = [row[:] for row in _IDENTITY_6]
        self._last_update = time.monotonic()
        self.location = (x, y, z)

    def get_prediction(self) -> tuple[float, float, float]:
        """Get predicted position without updating the filter."""
        if self._state is None or self._covariance is None:
            return self.location

        now = time.monotonic()
        dt = (now - self._last_update) if self._last_update else 0.1
        dt = max(dt, 0.001)

        f = _make_state_transition(dt)
        predicted = _mat_mul(f, self._state)
        return (predicted[0][0], predicted[1][0], predicted[2][0])

    def update(
        self,
        new_x: float,
        new_y: float,
        new_z: float,
        timestamp: float | None = None,
    ) -> tuple[float, float, float]:
        """Update filter with a new position measurement.

        Args:
            new_x: Measured x coordinate
            new_y: Measured y coordinate
            new_z: Measured z coordinate
            timestamp: Measurement timestamp (monotonic). Uses current time if None.

        Returns:
            Filtered (x, y, z) position.
        """
        now = timestamp if timestamp is not None else time.monotonic()

        # Initialize on first update
        if self._state is None or self._covariance is None:
            self.reset(new_x, new_y, new_z)
            self._last_update = now
            return self.location

        # Calculate time delta
        dt = (now - self._last_update) if self._last_update else 0.1
        self._last_update = now
        dt = max(dt, 0.001)  # Minimum 1ms

        settings = self.settings

        # Check if proposed move exceeds max velocity
        distance_to_new = math.sqrt(
            (self.location[0] - new_x) ** 2
            + (self.location[1] - new_y) ** 2
            + (self.location[2] - new_z) ** 2
        )
        max_possible_distance = settings.max_velocity * dt

        # Dynamic measurement noise - increase if movement is implausible
        dynamic_noise = settings.measurement_noise
        if distance_to_new > max_possible_distance and max_possible_distance > 0:
            excess_factor = distance_to_new / max_possible_distance
            dynamic_noise *= excess_factor * excess_factor

        # === PREDICT STEP ===
        f = _make_state_transition(dt)
        q = _make_process_noise(dt, settings.process_noise)

        # Predict state: x_pred = F * x
        self._state = _mat_mul(f, self._state)
        # Predict covariance: P_pred = F * P * F^T + Q
        f_t = _mat_transpose(f)
        self._covariance = _mat_add(_mat_mul(_mat_mul(f, self._covariance), f_t), q)

        # Constrain velocity after predict
        self._constrain_velocity()

        # === UPDATE STEP ===
        # Measurement noise covariance R (3x3 diagonal)
        r = [
            [dynamic_noise, 0.0, 0.0],
            [0.0, dynamic_noise, 0.0],
            [0.0, 0.0, dynamic_noise],
        ]

        # Measurement vector z (3x1)
        measurement = [[new_x], [new_y], [new_z]]

        # Innovation: y = z - H * x_pred
        predicted_measurement = _mat_mul(_H, self._state)
        innovation = _mat_sub(measurement, predicted_measurement)

        # Innovation covariance: S = H * P * H^T + R
        s = _mat_add(_mat_mul(_mat_mul(_H, self._covariance), _HT), r)

        # Kalman gain: K = P * H^T * S^-1
        s_inv = _inv3x3(s)
        k = _mat_mul(_mat_mul(self._covariance, _HT), s_inv)

        # Update state: x = x_pred + K * innovation
        self._state = _mat_add(self._state, _mat_mul(k, innovation))

        # Update covariance: P = (I - K * H) * P
        kh = _mat_mul(k, _H)
        i_minus_kh = _mat_sub([row[:] for row in _IDENTITY_6], kh)
        self._covariance = _mat_mul(i_minus_kh, self._covariance)

        # Constrain velocity after update
        self._constrain_velocity()

        # Extract filtered position
        self.location = (
            self._state[0][0],
            self._state[1][0],
            self._state[2][0],
        )
        return self.location

    def _constrain_velocity(self) -> None:
        """Clamp velocity magnitude to max_velocity, preserving direction."""
        if self._state is None:
            return

        vx = self._state[3][0]
        vy = self._state[4][0]
        vz = self._state[5][0]
        speed = math.sqrt(vx * vx + vy * vy + vz * vz)

        if speed > self.settings.max_velocity and speed > 0:
            scale = self.settings.max_velocity / speed
            self._state[3][0] = vx * scale
            self._state[4][0] = vy * scale
            self._state[5][0] = vz * scale
