"""
Continuous-time-propagated, discrete-measurement-updated Kalman filter
(matches the Euler-integrated prediction step used throughout the
original experiments, rather than a matrix-exponential discretization).
"""

import numpy as np


def make_step(A, B, C, Q, R, P0=None):
    """
    Return an observer_step(xhat, y_meas, u, dt, extra) callable.

    `extra` carries the error covariance P between calls; pass
    `observer_extra=P0` (or None, defaulting to identity) when calling
    simulate.run_simulation.
    """
    n = A.shape[0]
    P_init = np.eye(n) if P0 is None else np.asarray(P0, dtype=float)

    def step(xhat, y_meas, u, dt, extra):
        P = P_init.copy() if extra is None else extra

        # Prediction (Euler-integrated, matching the plant integration)
        xhat_pred = xhat + (A @ xhat + B * u) * dt
        P_pred = A @ P @ A.T + Q

        # Measurement update
        S = C @ P_pred @ C.T + R
        K = P_pred @ C.T @ np.linalg.inv(S)
        xhat_next = xhat_pred + K @ (y_meas - C @ xhat_pred)
        P_next = (np.eye(n) - K @ C) @ P_pred

        return xhat_next, P_next
    return step
