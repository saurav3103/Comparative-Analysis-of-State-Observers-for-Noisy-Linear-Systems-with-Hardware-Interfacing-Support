"""
System definitions for the 2nd-order LTI observer experiments.

Two system variants are provided:
    - `underdamped_system()`   -> A = [[0, 1], [-2, -3]]
    - `integrator_damped_system()` -> A = [[0, 1], [0, -3]]

Both share B = [[0], [1]], C = [[1, 0]], D = [[0]].
"""

import numpy as np


def underdamped_system():
    """Return (A, B, C, D) for the underdamped 2nd-order system."""
    A = np.array([[0, 1],
                  [-2, -3]], dtype=float)
    B = np.array([[0],
                  [1]], dtype=float)
    C = np.array([[1, 0]], dtype=float)
    D = np.array([[0]], dtype=float)
    return A, B, C, D


def integrator_damped_system():
    """Return (A, B, C, D) for the integrator-damped 2nd-order system."""
    A = np.array([[0, 1],
                  [0, -3]], dtype=float)
    B = np.array([[0],
                  [1]], dtype=float)
    C = np.array([[1, 0]], dtype=float)
    D = np.array([[0]], dtype=float)
    return A, B, C, D


def step_function(t, t_on=0.0):
    """
    Unit step input, active for t >= t_on.

    Works for both scalar `t` (used inside simulation loops) and
    array-like `t` (used for plotting / solve_ivp evaluation).
    """
    if np.isscalar(t):
        return 1.0 if t >= t_on else 0.0
    return np.where(np.asarray(t) < t_on, 0, 1)


def state_space_ode(t, x, A, B, u_func):
    """
    State-space derivative function, compatible with scipy.integrate.solve_ivp.

    x: flattened state vector
    u_func: callable u_func(t) -> scalar input
    """
    x = np.asarray(x).reshape(-1, 1)
    dxdt = A @ x + B * u_func(t)
    return dxdt.flatten()


def measurement(C, D, x, u):
    """Compute ideal output y = Cx + Du for a single state/input pair."""
    x = np.asarray(x).reshape(-1, 1)
    return (C @ x + D * u).flatten()
