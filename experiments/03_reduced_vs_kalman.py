"""
Experiment 3: Reduced-order observer vs Kalman filter, with disturbance.

Uses the integrator-damped system with a sinusoidal disturbance injected
into x2's dynamics, testing how each estimator handles an unmodeled
disturbance on top of measurement noise.

Run: python experiments/03_reduced_vs_kalman.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.system import integrator_damped_system, step_function
from src.simulate import run_simulation
from src.observers import reduced_order, kalman
from src.utils.metrics import compute_mse, compute_convergence_time, summarize
from src.utils.plotting import plot_comparison

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

    # --- Reduced-order ---
    red_step = reduced_order.make_step()
    t_eval, x_true, x_red = run_simulation(
        A, B, C, u_func, red_step, dt=0.01, t_final=10.0, noise_std=0.05,
        disturbance_amp=0.3, disturbance_freq=0.5, seed=0,
    )

    # --- Kalman ---
    Q = np.array([[1e-5, 0], [0, 1e-3]])
    R = np.array([[0.05 ** 2]])
    kf_step = kalman.make_step(A, B, C, Q, R)
    _, x_true_kf, x_kf = run_simulation(
        A, B, C, u_func, kf_step, dt=0.01, t_final=10.0, noise_std=0.05,
        disturbance_amp=0.3, disturbance_freq=0.5, seed=0, observer_extra=np.eye(2),
    )

    mse_red = compute_mse(x_true, x_red)
    mse_kf = compute_mse(x_true_kf, x_kf)
    print(f"Reduced-order MSE: x2 = {mse_red[1]:.6f}")
    print(f"Kalman MSE:        x2 = {mse_kf[1]:.6f}")

    conv_red = compute_convergence_time(t_eval, x_true[:, 1], x_red[:, 1])
    if conv_red is not None:
        print(f"Reduced-order observer converges in ≈ {conv_red:.2f} s")
    else:
        print("Reduced-order observer did not converge within the threshold.")

    fig = plot_comparison(
        t_eval, x_true, {"reduced-order": x_red, "kalman": x_kf},
        title="Reduced-Order vs Kalman under Disturbance (x2)",
        state_index=1, state_label="x2",
    )
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, "03_reduced_vs_kalman_x2.png"), dpi=150)

    append_summary([
        summarize(t_eval, x_true, x_red, "reduced_order"),
        summarize(t_eval, x_true_kf, x_kf, "kalman_reduced_system"),
    ])

    import matplotlib.pyplot as plt
    plt.show()


if __name__ == "__main__":
    main()
