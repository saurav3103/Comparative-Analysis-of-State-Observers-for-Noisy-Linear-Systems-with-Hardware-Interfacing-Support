"""
Experiment 2: Luenberger vs Kalman filter, full-order, no disturbance.

Compares a pole-placed Luenberger observer against a Kalman filter on
the underdamped system under measurement noise only. Reports per-state
MSE and appends results to results/mse_summary.csv.

Run: python experiments/02_luenberger_vs_kalman.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from src.system import underdamped_system, step_function
from src.simulate import run_simulation
from src.observers import luenberger, kalman
from src.utils.metrics import compute_mse, summarize
from src.utils.plotting import plot_comparison

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "mse_summary.csv")


def append_summary(rows):
    df_new = pd.DataFrame(rows)
    if os.path.exists(SUMMARY_CSV):
        df_old = pd.read_csv(SUMMARY_CSV)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(SUMMARY_CSV, index=False)
    print(f"Updated: {SUMMARY_CSV}")


def main():
    A, B, C, D = underdamped_system()
    u_func = lambda t: step_function(t, t_on=0.0)

    # --- Luenberger ---
    L = luenberger.design_gain(A, C, [-5, -6])
    luen_step = luenberger.make_step(A, B, C, L)
    t_eval, x_true, x_luen = run_simulation(
        A, B, C, u_func, luen_step, dt=0.01, t_final=10.0, noise_std=0.05, seed=42,
    )

    # --- Kalman ---
    Q = np.array([[1e-5, 0], [0, 1e-4]])
    R = np.array([[0.05 ** 2]])
    kf_step = kalman.make_step(A, B, C, Q, R)
    _, x_true_kf, x_kf = run_simulation(
        A, B, C, u_func, kf_step, dt=0.01, t_final=10.0, noise_std=0.05, seed=42,
        observer_extra=np.eye(2),
    )

    mse_luen = compute_mse(x_true, x_luen)
    mse_kf = compute_mse(x_true_kf, x_kf)
    print(f"Luenberger MSE: x1={mse_luen[0]:.5f}, x2={mse_luen[1]:.5f}")
    print(f"Kalman MSE:     x1={mse_kf[0]:.5f}, x2={mse_kf[1]:.5f}")

    fig = plot_comparison(
        t_eval, x_true, {"Luenberger": x_luen, "Kalman": x_kf},
        title="Luenberger vs Kalman (x1)", state_index=0, state_label="x1",
    )
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, "02_luenberger_vs_kalman_x1.png"), dpi=150)

    fig2 = plot_comparison(
        t_eval, x_true, {"Luenberger": x_luen, "Kalman": x_kf},
        title="Luenberger vs Kalman (x2)", state_index=1, state_label="x2",
    )
    fig2.savefig(os.path.join(FIG_DIR, "02_luenberger_vs_kalman_x2.png"), dpi=150)

    append_summary([
        summarize(t_eval, x_true, x_luen, "luenberger"),
        summarize(t_eval, x_true_kf, x_kf, "kalman"),
    ])

    import matplotlib.pyplot as plt
    plt.show()


if __name__ == "__main__":
    main()
