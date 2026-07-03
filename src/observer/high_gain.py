"""
High-gain observer and finite-time (sign-based) observer.

Both use the same generic full-order structure as the Luenberger
observer but with a differently-computed correction term:

    High-gain:    xhat_dot = A@xhat + B*u + [lambda_1; lambda_2] * (y - C@xhat)
    Finite-time:  xhat_dot = A@xhat + B*u + gain * sign(y - C@xhat)
"""

import numpy as np


def make_high_gain_step(A, B, C, lambda_1=30.0, lambda_2=50.0):
    """Fixed high-gain injection observer."""
    gain = np.array([[lambda_1], [lambda_2]])

    def step(xhat, y_meas, u, dt, extra):
        error = y_meas - C @ xhat
        dxhat = A @ xhat + B * u + gain @ error
        return xhat + dxhat * dt, extra
    return step


def make_finite_time_step(A, B, C, gain=100.0):
    """Finite-time observer using sign-based error injection."""
    def step(xhat, y_meas, u, dt, extra):
        error = y_meas - C @ xhat
        dxhat = A @ xhat + B * u + gain * np.sign(error)
        return xhat + dxhat * dt, extra
    return step
