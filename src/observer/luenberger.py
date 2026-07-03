"""
Full-order Luenberger observer via pole placement, plus a deadbeat
variant (same structure, faster poles).
"""

import numpy as np
from scipy.signal import place_poles


def design_gain(A, C, poles):
    """
    Compute the Luenberger observer gain L such that (A - LC) has
    eigenvalues at `poles`.

    Uses duality: place_poles on (A.T, C.T) then transposes the result.
    """
    result = place_poles(A.T, C.T, poles)
    return result.gain_matrix.T


def make_step(A, B, C, L):
    """
    Return an observer_step(xhat, y_meas, u, dt, extra) callable compatible
    with simulate.run_simulation, using Euler integration of the observer ODE:

        xhat_dot = A @ xhat + B*u + L @ (y_meas - C @ xhat)
    """
    def step(xhat, y_meas, u, dt, extra):
        dxhat = A @ xhat + B * u + L @ (y_meas - C @ xhat)
        return xhat + dxhat * dt, extra
    return step


def deadbeat_step_factory(A, B, C, poles=(-8, -9)):
    """Deadbeat observer: same structure as Luenberger, faster poles."""
    L = design_gain(A, C, list(poles))
    return make_step(A, B, C, L)
