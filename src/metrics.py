"""
Error metrics for comparing observer performance: MSE and convergence time.
"""

import numpy as np


def compute_mse(x_true, x_hat, axis=0):
    """
    Mean squared error per state.

    x_true, x_hat: (N, n) arrays. Returns an (n,) array of per-state MSE.
    """
    error = np.asarray(x_true) - np.asarray(x_hat)
    return np.mean(error ** 2, axis=axis)


def compute_convergence_time(t_eval, x_true, x_hat, threshold=0.01,
                              start_time=1.0, state_index=None):
    """
    Time at which |error| first drops below `threshold` and stays there,
    ignoring the initial transient before `start_time`.

    If state_index is None, x_true/x_hat are treated as 1-D (single state).
    Returns the convergence time, or None if the threshold is never met.
    """
    t_eval = np.asarray(t_eval)
    if state_index is not None:
        error = np.abs(np.asarray(x_true)[:, state_index] - np.asarray(x_hat)[:, state_index])
    else:
        error = np.abs(np.asarray(x_true) - np.asarray(x_hat))

    start_index = int(np.searchsorted(t_eval, start_time))
    below = np.where(error[start_index:] < threshold)[0]

    if below.size == 0:
        return None
    return t_eval[start_index + below[0]]


def summarize(t_eval, x_true, x_hat, label, threshold=0.01, start_time=1.0):
    """
    Convenience wrapper: returns a dict with per-state MSE and per-state
    convergence time, suitable for accumulating into results/mse_summary.csv.
    """
    mse = compute_mse(x_true, x_hat)
    n = x_true.shape[1]
    conv = [
        compute_convergence_time(t_eval, x_true, x_hat, threshold, start_time, state_index=i)
        for i in range(n)
    ]
    return {
        "observer": label,
        **{f"mse_x{i+1}": mse[i] for i in range(n)},
        **{f"convergence_x{i+1}": conv[i] for i in range(n)},
    }
