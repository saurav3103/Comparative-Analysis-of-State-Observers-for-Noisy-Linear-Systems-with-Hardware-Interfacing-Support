"""
Reduced-order observer for the integrator-damped system
    A = [[0, 1], [0, -3]], B = [[0], [1]], C = [[1, 0]]

x1 is measured directly (assumed noisy but available); only x2 is
estimated via a first-order observer:

    x2_hat_dot = -3*x2_hat - 2*x1_meas + u

Note: this observer's dynamics are hard-coded to the integrator-damped
system's structure. For a different A/B/C, re-derive the reduced-order
observer equation before reusing this module.
"""

import numpy as np


def make_step():
    """
    Return an observer_step(xhat, y_meas, u, dt, extra) callable.

    `xhat` is kept as a 2-vector for interface compatibility with the
    other observers: xhat[0] is set directly to the x1 measurement,
    xhat[1] is the estimated x2.
    """
    def step(xhat, y_meas, u, dt, extra):
        x1_meas = y_meas[0, 0]
        x2_hat = xhat[1, 0]

        dx2_hat = -3 * x2_hat - 2 * x1_meas + u
        x2_hat_next = x2_hat + dx2_hat * dt

        xhat_next = np.array([[x1_meas], [x2_hat_next]])
        return xhat_next, extra
    return step
