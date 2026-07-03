"""
Shared simulation utilities: true-system propagation, noisy measurement
generation, and disturbance injection. Used by every experiment script so
each observer comparison runs against an identical plant simulation.
"""

import numpy as np


def make_time_vector(t_final=10.0, dt=0.01):
    """Return the evaluation time vector for a simulation run."""
    return np.arange(0, t_final, dt)


def sinusoidal_disturbance(t, amplitude=0.2, freq_hz=0.5):
    """Sinusoidal disturbance injected into a state derivative."""
    return amplitude * np.sin(2 * np.pi * freq_hz * t)


def step_true_state(x, A, B, u, dt, disturbance=0.0, disturbance_state=1):
    """
    Advance the true system state by one Euler integration step.

    disturbance: scalar disturbance value added to the derivative of
                 `disturbance_state` (default: index 1, i.e. x2).
    """
    dx = A @ x + B * u
    dx[disturbance_state, 0] += disturbance
    return x + dx * dt


def noisy_measurement(C, x, noise_std, rng=None):
    """Return C @ x corrupted with zero-mean Gaussian noise."""
    y_true = C @ x
    noise_fn = rng.normal if rng is not None else np.random.normal
    noise = noise_fn(0, noise_std, size=y_true.shape)
    return y_true + noise


def run_simulation(A, B, C, u_func, observer_step, dt=0.01, t_final=10.0,
                    noise_std=0.05, disturbance_amp=0.0, disturbance_freq=0.5,
                    disturbance_state=1, seed=0, x0=None, xhat0=None,
                    observer_extra=None):
    """
    Generic simulation driver: propagates the true plant and a single
    observer in lockstep, returning true/estimated state histories.

    observer_step: callable with signature
        xhat_next, extra = observer_step(xhat, y_meas, u, dt, extra)
    where `extra` carries any observer-specific persistent state
    (e.g. Kalman covariance P). Pass `observer_extra` to seed it.

    Returns
    -------
    t_eval : (N,) ndarray
    x_true : (N, n) ndarray
    x_hat  : (N, n) ndarray
    """
    rng = np.random.default_rng(seed)
    t_eval = make_time_vector(t_final, dt)

    n = A.shape[0]
    x = np.zeros((n, 1)) if x0 is None else np.asarray(x0, dtype=float).reshape(n, 1)
    xhat = np.zeros((n, 1)) if xhat0 is None else np.asarray(xhat0, dtype=float).reshape(n, 1)
    extra = observer_extra

    x_true = np.zeros((len(t_eval), n))
    x_hat = np.zeros((len(t_eval), n))

    for i, t in enumerate(t_eval):
        u = u_func(t)

        disturbance = sinusoidal_disturbance(t, disturbance_amp, disturbance_freq) \
            if disturbance_amp else 0.0
        x = step_true_state(x, A, B, u, dt, disturbance, disturbance_state)
        y_meas = noisy_measurement(C, x, noise_std, rng)

        xhat, extra = observer_step(xhat, y_meas, u, dt, extra)

        x_true[i] = x.flatten()
        x_hat[i] = xhat.flatten()

    return t_eval, x_true, x_hat
