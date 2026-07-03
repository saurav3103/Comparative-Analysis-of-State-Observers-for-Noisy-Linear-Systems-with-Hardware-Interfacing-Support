"""
Interactive observer comparison dashboard.

Replaces the original ipywidgets/Colab interact() panel with a Streamlit
app: pick an observer type, tune its parameters and the plant's A/B/C
matrices, and see the state estimate vs. true state plotted live. An
optional "Live sensor" mode reads real measurements from a serial device
instead of simulating the plant (see app/serial_interface.py).

Run: streamlit run app/streamlit_app.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.signal import place_poles

from src.system import step_function
from src.observers import luenberger, reduced_order, kalman, sliding_mode, high_gain
from src.utils.metrics import compute_mse
from app.serial_interface import try_serial_connection, get_sensor_data, close_connection

st.set_page_config(page_title="LTI Observer Comparison", layout="wide")
st.title("LTI Observer Comparison Dashboard")

OBSERVER_OPTIONS = [
    "Luenberger", "Kalman", "SMO", "High-Gain",
    "Reduced-Order", "Deadbeat", "Finite-Time",
]

# ---------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Observer")
    observer = st.selectbox("Type", OBSERVER_OPTIONS)

    st.header("Noise & Disturbance")
    noise_std = st.slider("Measurement noise σ", 0.0, 0.1, 0.05, 0.01)
    disturbance_amp = st.slider("Disturbance amplitude", 0.0, 1.0, 0.2, 0.05)

    st.header("Observer Gains")
    lambda_1 = st.slider("λ₁ (SMO / High-Gain)", 0, 100, 30, 5)
    lambda_2 = st.slider("λ₂ (SMO / High-Gain)", 0, 100, 50, 5)
    pole1 = st.slider("Luenberger pole 1", -20.0, -0.1, -5.0, 0.1)
    pole2 = st.slider("Luenberger pole 2", -20.0, -0.1, -6.0, 0.1)

    st.header("Kalman Tuning")
    q1 = st.number_input("Q[0,0]", value=1e-5, format="%.6f")
    q2 = st.number_input("Q[1,1]", value=1e-4, format="%.6f")
    r = st.number_input("R", value=2.5e-3, format="%.6f")

    st.header("Plant Matrices")
    matrix_input_mode = st.radio("A, B, C source", ["Manual", "Live sensor (serial)"])
    A_text = st.text_area("A", value="[[0, 1], [0, -3]]")
    B_text = st.text_area("B", value="[[0], [1]]")
    C_text = st.text_area("C", value="[[1, 0]]")

    run_button = st.button("Run simulation", type="primary")


# ---------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------
def build_observer_step(observer_name, A, B, C):
    """Return (observer_step, initial_extra) for the selected observer."""
    if observer_name == "Luenberger":
        L = luenberger.design_gain(A, C, [pole1, pole2])
        return luenberger.make_step(A, B, C, L), None
    if observer_name == "Deadbeat":
        return luenberger.deadbeat_step_factory(A, B, C, poles=(-8, -9)), None
    if observer_name == "Reduced-Order":
        return reduced_order.make_step(), None
    if observer_name == "Kalman":
        Q = np.array([[q1, 0], [0, q2]])
        R = np.array([[r]])
        return kalman.make_step(A, B, C, Q, R), np.eye(A.shape[0])
    if observer_name == "SMO":
        return sliding_mode.make_step(lambda_1, lambda_2, sat_bound=0.01), None
    if observer_name == "High-Gain":
        return high_gain.make_high_gain_step(A, B, C, lambda_1, lambda_2), None
    if observer_name == "Finite-Time":
        return high_gain.make_finite_time_step(A, B, C, gain=100.0), None
    raise ValueError(f"Unknown observer: {observer_name}")


def run_dashboard_simulation(A, B, C, use_serial, dt=0.01, t_final=10.0):
    t_eval = np.arange(0, t_final, dt)
    n = A.shape[0]

    x = np.zeros((n, 1))
    xhat = np.zeros((n, 1))
    obs_step, extra = build_observer_step(observer, A, B, C)

    ser = try_serial_connection() if use_serial else None
    if use_serial and ser is None:
        st.error("No serial device connected. Check the port in app/serial_interface.py.")
        return None, None, None

    x_true, x_est = [], []
    rng = np.random.default_rng(0)

    try:
        for t in t_eval:
            u = step_function(t, t_on=1.0)

            if not use_serial:
                dx = A @ x + B * u
                dx[min(1, n - 1), 0] += disturbance_amp * np.sin(2 * np.pi * 0.5 * t)
                x = x + dx * dt
                y_meas = C @ x + rng.normal(0, noise_std, size=(C.shape[0], 1))
                x_true.append(x.flatten())
            else:
                val = get_sensor_data(ser)
                if val is None:
                    continue
                y_meas = np.array([[val]])
                x_true.append([np.nan] * n)

            xhat, extra = obs_step(xhat, y_meas, u, dt, extra)
            x_est.append(xhat.flatten())
    finally:
        if ser is not None:
            close_connection(ser)

    return t_eval[:len(x_est)], np.array(x_true), np.array(x_est)


# ---------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------
if run_button:
    try:
        A = np.array(eval(A_text), dtype=float)
        B = np.array(eval(B_text), dtype=float)
        C = np.array(eval(C_text), dtype=float)
    except Exception as exc:
        st.error(f"Invalid matrix input: {exc}")
        st.stop()

    use_serial = matrix_input_mode == "Live sensor (serial)"
    t_eval, x_true, x_est = run_dashboard_simulation(A, B, C, use_serial)

    if t_eval is None:
        st.stop()

    n = A.shape[0]
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(t_eval, x_est[:, i], label=f"x{i+1} (estimated)")
        if not use_serial:
            ax.plot(t_eval, x_true[:, i], label=f"x{i+1} (true)")
        ax.set_ylabel(f"x{i+1}")
        ax.legend()
        ax.grid(True)
    axes[-1].set_xlabel("Time (s)")

    if use_serial:
        fig.suptitle(f"{observer} Observer | Live sensor input")
    else:
        mse = compute_mse(x_true, x_est)
        mse_str = ", ".join(f"x{i+1}={mse[i]:.4f}" for i in range(n))
        fig.suptitle(f"{observer} Observer | MSE: {mse_str}")
        st.metric("Overall MSE", f"{np.mean(mse):.5f}")

    plt.tight_layout()
    st.pyplot(fig)
else:
    st.info("Configure the observer and plant in the sidebar, then click **Run simulation**.")
