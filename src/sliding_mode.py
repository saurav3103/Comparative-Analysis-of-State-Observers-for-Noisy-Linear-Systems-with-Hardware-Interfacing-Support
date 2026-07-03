"""
Sliding Mode Observer (SMO) for the integrator-damped system
    A = [[0, 1], [0, -3]], B = [[0], [1]], C = [[1, 0]]

Uses a saturation function in place of a hard sign() to reduce chattering
near the sliding surface.

Note: like reduced_order.py, the observer dynamics below are specific to
this system's structure (x2_dot = -3*x2 - 2*x1 + u); re-derive for other
A/B/C matrices.
"""

import numpy as np


def saturate(value, bound):
    """Saturation function: clip(value / bound, -1, 1)."""
    return np.clip(value / bound, -1, 1)


def make_step(lambda_1=30.0, lambda_2=50.0, sat_bound=0.01):
    """
    Return an observer_step(xhat, y_meas, u, dt, extra) callable.

    lambda_1, lambda_2: sliding-mode injection gains
    sat_bound: saturation boundary layer width
    """
    def step(xhat, y_meas, u, dt, extra):
        y_est = xhat[0, 0]  # C @ xhat, with C = [1, 0]
        s = y_meas[0, 0] - y_est
        sat = saturate(s, sat_bound)

        dxhat = np.array([
            [xhat[1, 0] + lambda_1 * sat],
            [-3 * xhat[1, 0] - 2 * y_meas[0, 0] + u + lambda_2 * sat]
        ])
        return xhat + dxhat * dt, extra
    return step
