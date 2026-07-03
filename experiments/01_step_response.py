"""
Experiment 1: Open-loop step response.

Simulates the underdamped 2nd-order system's response to a unit step
input via scipy.integrate.solve_ivp, then overlays a noisy measurement
to visualize what an observer would actually see.

Run: python experiments/01_step_response.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from src.system import underdamped_system, step_function, state_space_ode, measurement

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "figures")


def main():
    A, B, C, D = underdamped_system()

    t_span = (0, 10)
    t_eval = np.linspace(*t_span, 500)
    x0 = [0, 0]

    sol = solve_ivp(
        lambda t, x: state_space_ode(t, x, A, B, step_function),
        t_span, x0, t_eval=t_eval,
    )
    X = sol.y.T  # (500, 2)

    Y_clean = np.array([measurement(C, D, x, step_function(t)) for x, t in zip(X, t_eval)])

    noise_std = 0.05
    rng = np.random.default_rng(0)
    Y_noisy = Y_clean + rng.normal(0, noise_std, size=Y_clean.shape)

    plt.figure(figsize=(10, 5))
    plt.plot(t_eval, X[:, 0], label='x1 (position)')
    plt.plot(t_eval, X[:, 1], label='x2 (velocity)')
    plt.plot(t_eval, Y_noisy.flatten(), '--', label='y (noisy output)', color='orange')
    plt.xlabel('Time (s)')
    plt.ylabel('States / Output')
    plt.title('State-Space Step Response with Noisy Output')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "01_step_response.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
