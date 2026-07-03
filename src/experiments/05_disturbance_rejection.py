"""
Experiment 5: Disturbance rejection sweep across all observers.

Runs every observer implemented in src/observers against the same
disturbed, noisy plant and ranks them by MSE. This is the "all-in-one"
comparison used to populate the full results/mse_summary.csv table
referenced in the README.

Run: python experiments/05_disturbance_rejection.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from src.system import integrator_damped_system, step_function
from src.simulate import run_simulation
from src.observers import luenberger, reduced_order, kalman, sliding_mode, high_gain
from src.utils.metrics import compute_mse, summarize
from src.utils.plotting import plot_comparison

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "mse_summary.csv")


def build_observers(A, B, C):
    """Return {name: (observer_step, observer_extra)} for every observer type."""
    L = luenberger.design_gain(A, C, [-5, -6])
    L_deadbeat = luenberger.design_gain(A, C, [-8, -9])
    Q = np.array([[1e-5, 0], [0, 1e-3]])
    R = np.array([[0.05 ** 2]])

    return {
        "luenberger": (luenberger.make_step(A, B, C, L), None),
        "deadbeat": (luenberger.make_step(A, B, C, L_deadbeat), None),
        "reduced_order": (reduced_order.make_step(), None),
        "kalman": (kalman.make_step(A, B, C, Q, R), np.eye(2)),
        "smo": (sliding_mode.make_step(lambda_1=30.0, lambda_2=50.0, sat_bound=0.01), None),
        "high_gain": (high_gain.make_high_gain_step(A, B, C, lambda_1=30.0, lambda_2=50.0), None),
        "finite_time": (high_gain.make_finite_time_step(A, B, C, gain=100.0), None),
    }


def main():
    A, B, C, D = integrator_damped_system()
    u_func = lambda t: step_function(t, t_on=1.0)

    observers = build_observers(A, B, C)
    estimates = {}
    rows = []
    t_eval = None

    for name, (obs_step, extra) in observers.items():
        t_eval, x_true, x_hat = run_simulation(
            A, B, C, u_func, obs_step, dt=0.01, t_final=10.0, noise_std=0.05,
            disturbance_amp=0.3, disturbance_freq=0.5, seed=0, observer_extra=extra,
        )
        estimates[name] = x_hat
        rows.append(summarize(t_eval, x_true, x_hat, name))
        mse = compute_mse(x_true, x_hat)
        print(f"{name:15s} MSE: x1={mse[0]:.6f}, x2={mse[1]:.6f}")

    ranked = sorted(rows, key=lambda r: r["mse_x2"])
    print("\nRanked by x2 MSE (best to worst):")
    for r in ranked:
        print(f"  {r['observer']:15s} mse_x2={r['mse_x2']:.6f}")

    # Re-run true trajectory once more for the comparison plot (any observer's x_true works)
    _, x_true_ref, _ = run_simulation(
        A, B, C, u_func, observers["luenberger"][0], dt=0.01, t_final=10.0,
        noise_std=0.05, disturbance_amp=0.3, disturbance_freq=0.5, seed=0,
    )

    fig = plot_comparison(
        t_eval, x_true_ref, estimates,
        title="All Observers under Disturbance (x2)", state_index=1, state_label="x2",
    )
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, "05_disturbance_rejection_all_x2.png"), dpi=150)

    df_new = pd.DataFrame(rows)
    if os.path.exists(SUMMARY_CSV):
        df_old = pd.read_csv(SUMMARY_CSV)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(SUMMARY_CSV, index=False)
    print(f"\nUpdated: {SUMMARY_CSV}")

    import matplotlib.pyplot as plt
    plt.show()


if __name__ == "__main__":
    main()
