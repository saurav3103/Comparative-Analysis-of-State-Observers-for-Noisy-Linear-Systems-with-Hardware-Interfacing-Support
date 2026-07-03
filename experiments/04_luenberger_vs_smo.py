"""
Experiment 4: Luenberger vs Sliding Mode Observer (SMO), with disturbance.

Compares a full-order Luenberger observer against a sliding-mode
observer on the integrator-damped system, under both measurement noise
and a sinusoidal disturbance on x2. SMO uses a saturation boundary layer
to reduce chattering.

Run: python experiments/04_luenberger_vs_smo.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.system import integrator_damped_system, step_function
from src.simulate import run_simulation
from src.observers import luenberger, sliding_mode
from src.utils.metrics import compute_mse, summarize
from src.utils.plotting import plot_states, plot_comparison

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "mse_summary.csv")


def append_summary(rows):
    import pandas as pd
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
    A, B, C, D = integrator_damped_system()
    u_func = lambda t: step_function(t, t_on=1.0)

    # --- Luenberger ---
    L = luenberger.design_gain(A, C, [-5, -6])
    luen_step = luenberger.make_step(A, B, C, L)
    t_eval, x_true, x_luen = run_simulation(
        A, B, C, u_func, luen_step, dt=0.01, t_final=10.0, noise_std=0.05,
        disturbance_amp=0.2, disturbance_freq=0.5, seed=0,
    )

    # --- SMO ---
    smo_step = sliding_mode.make_step(lambda_1=30.0, lambda_2=50.0, sat_bound=0.01)
    _, x_true_smo, x_smo = run_simulation(
        A, B, C, u_func, smo_step, dt=0.01, t_final=10.0, noise_std=0.05,
        disturbance_amp=0.2, disturbance_freq=0.5, seed=0,
    )

    mse_luen = compute_mse(x_true, x_luen)
    mse_smo = compute_mse(x_true_smo, x_smo)
    print(f"Luenberger MSE: x1={mse_luen[0]:.6f}, x2={mse_luen[1]:.6f}")
    print(f"SMO MSE:        x1={mse_smo[0]:.6f}, x2={mse_smo[1]:.6f}")

    fig = plot_states(t_eval, x_true, x_luen, title="Luenberger Observer", labels=["x1", "x2"])
    fig2 = plot_states(t_eval, x_true_smo, x_smo, title="SMO", labels=["x1", "x2"])

    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, "04_luenberger_states.png"), dpi=150)
    fig2.savefig(os.path.join(FIG_DIR, "04_smo_states.png"), dpi=150)

    fig3 = plot_comparison(
        t_eval, x_true, {"Luenberger": x_luen, "SMO": x_smo},
        title="Luenberger vs SMO (x2)", state_index=1, state_label="x2",
    )
    fig3.savefig(os.path.join(FIG_DIR, "04_luenberger_vs_smo_x2.png"), dpi=150)

    append_summary([
        summarize(t_eval, x_true, x_luen, "luenberger"),
        summarize(t_eval, x_true_smo, x_smo, "smo"),
    ])

    import matplotlib.pyplot as plt
    plt.show()


if __name__ == "__main__":
    main()
