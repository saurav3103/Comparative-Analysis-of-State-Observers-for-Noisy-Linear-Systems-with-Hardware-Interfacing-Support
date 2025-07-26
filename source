### Utilties
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

##Define matrix for LTI Systems
A = np.matrix([[0,1],[2,3]], int)
B = np.matrix([[0],[1]],int)
C = np.matrix([1,0],int)
D = 0
print(A) , print(B) , print(C) , print(D)

##Step signal
t = np.linspace(-10,10,10000)
step =np.where(t<0,0,1)
plt.plot(t,step)
plt.show()

"""### Defining LTI System"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define system matrices (example: second-order system)
A = np.array([[0, 1],
              [-2, -3]])
B = np.array([[0],
              [1]])
C = np.array([[1, 0]])
D = np.array([[0]])

# Define step input function
def step_function(t):
    return np.where(t < 0, 0, 1)

# Define state-space differential equations
def state_space(t, x):
    x = x.reshape(-1, 1)
    dxdt = A @ x + B * step_function(t)
    return dxdt.flatten()

# Initial condition
x0 = [0, 0]  # x1(0) = 0, x2(0) = 0

# Time span and evaluation points
t_span = (0, 10)
t_eval = np.linspace(*t_span, 500)

# Solve the ODE
sol = solve_ivp(state_space, t_span, x0, t_eval=t_eval)
X = sol.y.T  # shape: (500, 2)

# Compute ideal output y = Cx + Du
Y_clean = np.array([C @ x.reshape(-1, 1) + D * step_function(t) for x, t in zip(X, t_eval)])

# Add Gaussian noise to simulate non-ideal measurement
noise_std = 0.05  # standard deviation of measurement noise
noise = np.random.normal(0, noise_std, size=Y_clean.shape)
Y_noisy = Y_clean + noise

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(t_eval, X[:, 0], label='x1 (position)')
plt.plot(t_eval, X[:, 1], label='x2 (velocity)')
plt.plot(t_eval, Y_noisy.flatten(), label='y (noisy output)', linestyle='--', color='orange')
plt.xlabel('Time (s)')
plt.ylabel('States / Output')
plt.title('State-Space System Response with Noisy Output')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

"""###Luenberger Observer with Error Analysis"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import place_poles

# Set random seed for reproducibility
np.random.seed(42)

# Desired poles for fast convergence
desired_poles = [-5, -6]
L = place_poles(A.T, C.T, desired_poles).gain_matrix.T

# Noise characteristics
noise_std = 0.05  # Standard deviation of Gaussian noise

# Observer simulation
x_true = np.zeros((len(t_eval), 2))
x_hat = np.zeros((len(t_eval), 2))
x = np.array([[0], [0]])         # initial true state
xhat = np.array([[0], [0]])      # initial estimated state

dt = t_eval[1] - t_eval[0]       # fixed time step

for i, t in enumerate(t_eval):
    # Save current values
    x_true[i] = x.flatten()
    x_hat[i] = xhat.flatten()

    # True system output + measurement noise
    true_output = C @ x
    noise = np.random.normal(0, noise_std, size=(1, 1))  # shape (1,1)
    y_measured = true_output + noise

    # System update (Euler integration)
    dx = A @ x + B * step_function(t)
    x = x + dx * dt

    # Observer update
    dxhat = A @ xhat + B * step_function(t) + L @ (y_measured - C @ xhat)
    xhat = xhat + dxhat * dt

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(t_eval, x_true[:, 0], label='x1 (true)')
plt.plot(t_eval, x_hat[:, 0], '--', label='x1 (estimated)')
plt.plot(t_eval, x_true[:, 1], label='x2 (true)')
plt.plot(t_eval, x_hat[:, 1], '--', label='x2 (estimated)')
plt.xlabel('Time (s)')
plt.ylabel('States')
plt.title('Luenberger Observer with Noisy Measurements')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Compute estimation error
e = x_true - x_hat  # shape: (len(t_eval), 2)

plt.figure(figsize=(12, 6))

# Plot x1 and its estimation
plt.subplot(2, 1, 1)
plt.plot(t_eval, x_true[:, 0], label='x₁ (true)', linewidth=2)
plt.plot(t_eval, x_hat[:, 0], '--', label='x₁ (estimated)', linewidth=2)
plt.plot(t_eval, e[:, 0], ':', label='x₁ error', color='red', linewidth=1.5)
plt.ylabel('x₁ value')
plt.title('State x₁: True vs Estimated and Error')
plt.legend()
plt.grid(True)

# Plot x2 and its estimation
plt.subplot(2, 1, 2)
plt.plot(t_eval, x_true[:, 1], label='x₂ (true)', linewidth=2)
plt.plot(t_eval, x_hat[:, 1], '--', label='x₂ (estimated)', linewidth=2)
plt.plot(t_eval, e[:, 1], ':', label='x₂ error', color='red', linewidth=1.5)
plt.xlabel('Time (s)')
plt.ylabel('x₂ value')
plt.title('State x₂: True vs Estimated and Error')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

"""### Comparing Kalman and Luenberger Observer for defined LTI System"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import place_poles

# System matrices
A = np.array([[0, 1],
              [-2, -3]])
B = np.array([[0],
              [1]])
C = np.array([[1, 0]])
D = np.array([[0]])

# Time vector
dt = 0.01
t_eval = np.arange(0, 10, dt)

# Step input function
def step_function(t):
    return 1.0 if t >= 0 else 0.0

# Luenberger observer pole placement
desired_poles = [-5, -6]
L = place_poles(A.T, C.T, desired_poles).gain_matrix.T

# Noise characteristics
noise_std = 0.05
np.random.seed(42)

# Initial conditions
x = np.array([[0], [0]])        # true state
xhat = np.array([[0], [0]])     # Luenberger estimate
x_kf = np.array([[0], [0]])     # Kalman estimate
P = np.eye(2)                   # Kalman initial error covariance

Q = np.array([[1e-5, 0],        # Kalman process noise
              [0, 1e-4]])
R = np.array([[noise_std**2]])  # Kalman measurement noise

# Containers for all data
x_true = np.zeros((len(t_eval), 2))
x_hat = np.zeros((len(t_eval), 2))
x_kalman = np.zeros((len(t_eval), 2))

# Simulation
for i, t in enumerate(t_eval):
    # Save current state
    x_true[i] = x.flatten()
    x_hat[i] = xhat.flatten()
    x_kalman[i] = x_kf.flatten()

    # Measurement with noise
    true_output = C @ x
    noise = np.random.normal(0, noise_std, size=(1, 1))
    y_measured = true_output + noise

    # True system update
    dx = A @ x + B * step_function(t)
    x = x + dx * dt

    # Luenberger observer update
    dxhat = A @ xhat + B * step_function(t) + L @ (y_measured - C @ xhat)
    xhat = xhat + dxhat * dt

    # Kalman filter update
    # Prediction
    x_kf = x_kf + (A @ x_kf + B * step_function(t)) * dt
    P = A @ P @ A.T + Q

    # Measurement update
    S = C @ P @ C.T + R
    K = P @ C.T @ np.linalg.inv(S)
    x_kf = x_kf + K @ (y_measured - C @ x_kf)
    P = (np.eye(2) - K @ C) @ P

# ==========================
# 📈 Plotting
# ==========================

plt.figure(figsize=(12, 6))

# x1 plot
plt.subplot(2, 1, 1)
plt.plot(t_eval, x_true[:, 0], label='x₁ (true)', linewidth=2)
plt.plot(t_eval, x_hat[:, 0], '--', label='x₁ (Luenberger)', linewidth=2)
plt.plot(t_eval, x_kalman[:, 0], ':', label='x₁ (Kalman)', linewidth=2)
plt.ylabel('x₁ value')
plt.title('State x₁: True vs Estimated')
plt.legend()
plt.grid(True)

# x2 plot
plt.subplot(2, 1, 2)
plt.plot(t_eval, x_true[:, 1], label='x₂ (true)', linewidth=2)
plt.plot(t_eval, x_hat[:, 1], '--', label='x₂ (Luenberger)', linewidth=2)
plt.plot(t_eval, x_kalman[:, 1], ':', label='x₂ (Kalman)', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('x₂ value')
plt.title('State x₂: True vs Estimated')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ==========================
# 📊 Error Analysis
# ==========================

e_luenberger = x_true - x_hat
e_kalman = x_true - x_kalman

mse_luenberger = np.mean(e_luenberger**2, axis=0)
mse_kalman = np.mean(e_kalman**2, axis=0)

print(f"Luenberger MSE: x₁ = {mse_luenberger[0]:.5f}, x₂ = {mse_luenberger[1]:.5f}")
print(f"Kalman Filter MSE: x₁ = {mse_kalman[0]:.5f}, x₂ = {mse_kalman[1]:.5f}")

import numpy as np
import matplotlib.pyplot as plt

# System matrices
A = np.array([[0, 1],
              [0, -3]])
B = np.array([[0],
              [1]])

# Time settings
dt = 0.01
t_final = 10
t_eval = np.arange(0, t_final, dt)

# Step input
def step_function(t):
    return 1.0 if t >= 1 else 0.0

# Noise and disturbance settings
np.random.seed(0)
noise_std = 0.05                 # Measurement noise (x1)
disturbance_amplitude = 0.3     # Disturbance (affects x2 dynamics)

# Initial values
x = np.array([[0], [0]])          # true state
x2_hat = 0.0                      # initial estimate of x₂

x2_hat_list = []
x2_true_list = []
x1_meas_list = []

for t in t_eval:
    u = step_function(t)

    # True system evolution (with disturbance)
    dx = A @ x + B * u
    dx[1, 0] += disturbance_amplitude * np.sin(2 * np.pi * 0.5 * t)  # external disturbance in dx2
    x = x + dx * dt
    x1 = x[0, 0]
    x2 = x[1, 0]

    # Measurement with noise
    x1_meas = x1 + np.random.normal(0, noise_std)

    # Reduced-order observer update (uses noisy x1 measurement)
    dx2_hat = -3 * x2_hat - 2 * x1_meas + u
    x2_hat += dx2_hat * dt

    # Store values
    x1_meas_list.append(x1_meas)
    x2_true_list.append(x2)
    x2_hat_list.append(x2_hat)

# Plotting results
plt.figure(figsize=(10, 5))
plt.plot(t_eval, x2_true_list, label='x₂ (true)', linewidth=2)
plt.plot(t_eval, x2_hat_list, '--', label='x₂ (estimated, reduced-order)', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('x₂ value')
plt.title('Reduced-Order Observer under Noisy and Disturbed Conditions')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

"""### Time Complexity and Error Analysis of Full State and Reduced State Observers"""

e = np.abs(np.array(x2_true_list) - np.array(x2_hat_list))
threshold = 0.01

# Start checking only after 1 second to avoid early transient
start_index = int(1 / dt)
below_thresh_indices = np.where(e[start_index:] < threshold)[0]

if below_thresh_indices.size > 0:
    t_converge = t_eval[start_index + below_thresh_indices[0]]
    print(f"Reduced-order observer converges in ≈ {t_converge:.2f} s")
else:
    print("Reduced-order observer did not converge within the threshold.")

e2 = np.abs(np.array(x2_true_list) - np.array(x2_hat_list))
threshold = 0.01

# Start checking only after 1 second to avoid early transient
start_index = int(1 / dt)
below_thresh_indices = np.where(e2[start_index:] < threshold)[0]

if below_thresh_indices.size > 0:
    t_converge = t_eval[start_index + below_thresh_indices[0]]
    print(f"Reduced-order observer converges in ≈ {t_converge:.2f} s")
else:
    print("Reduced-order observer did not converge within the threshold.")

# Recalculate e using the true and estimated states from the full-order observer
e = x_true - x_hat

mse_full = np.mean(e**2, axis=0)
print(f"Full-order observer MSE: x1 = {mse_full[0]:.4f}, x2 = {mse_full[1]:.4f}")

mse_red = np.mean(e2**2)
print(f"Reduced-order observer MSE: x2 = {mse_red:.4f}")

"""### Use of Kalman Filter on Reduced State Observer"""

import numpy as np
import matplotlib.pyplot as plt

# System matrices
A = np.array([[0, 1],
              [0, -3]])
B = np.array([[0],
              [1]])
C = np.array([[1, 0]])
D = np.array([[0]])

# Time settings
dt = 0.01
t_final = 10
t_eval = np.arange(0, t_final, dt)

# Step input
def step_function(t):
    return 1.0 if t >= 1 else 0.0

# Noise and disturbance settings
np.random.seed(0)
noise_std = 0.05                 # Measurement noise (x1)
disturbance_amplitude = 0.3     # Disturbance (affects x2 dynamics)

# Initial values
x = np.array([[0], [0]])  # true state
x2_hat = 0.0              # reduced-order observer estimate

x2_hat_list = []
x2_true_list = []
x1_meas_list = []

# For Kalman filter
x_kalman = np.array([[0], [0]])  # initial KF state estimate
P = np.eye(2)                    # initial covariance
Q = np.array([[1e-5, 0],         # process noise covariance
              [0, 1e-3]])
R = np.array([[noise_std**2]])   # measurement noise covariance

x_kalman_list = []

for t in t_eval:
    u = step_function(t)

    # === True System ===
    dx = A @ x + B * u
    dx[1, 0] += disturbance_amplitude * np.sin(2 * np.pi * 0.5 * t)  # external disturbance in dx2
    x = x + dx * dt
    x1 = x[0, 0]
    x2 = x[1, 0]

    # === Measurement ===
    x1_meas = x1 + np.random.normal(0, noise_std)

    # === Reduced-Order Observer ===
    dx2_hat = -3 * x2_hat - 2 * x1_meas + u
    x2_hat += dx2_hat * dt

    # === Kalman Filter ===
    # Prediction
    x_kalman = x_kalman + (A @ x_kalman + B * u) * dt
    P = A @ P @ A.T + Q

    # Measurement update
    y = np.array([[x1_meas]])
    S = C @ P @ C.T + R
    K = P @ C.T @ np.linalg.inv(S)
    x_kalman = x_kalman + K @ (y - C @ x_kalman)
    P = (np.eye(2) - K @ C) @ P

    # === Store values ===
    x1_meas_list.append(x1_meas)
    x2_true_list.append(x2)
    x2_hat_list.append(x2_hat)
    x_kalman_list.append(x_kalman.flatten())

# === Convert Kalman output to numpy array ===
x_kalman_array = np.array(x_kalman_list)
x2_kalman = x_kalman_array[:, 1]

# === Plotting ===
plt.figure(figsize=(12, 6))
plt.plot(t_eval, x2_true_list, label='x₂ (true)', linewidth=2)
plt.plot(t_eval, x2_hat_list, '--', label='x₂ (reduced-order observer)', linewidth=2)
plt.plot(t_eval, x2_kalman, ':', label='x₂ (Kalman filter)', linewidth=2.2)
plt.xlabel('Time (s)')
plt.ylabel('x₂ value')
plt.title('Comparison: True vs Reduced-Order vs Kalman Filter (x₂)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Error Metrics ===
e_red = np.abs(np.array(x2_true_list) - np.array(x2_hat_list))
e_kalman = np.abs(np.array(x2_true_list) - x2_kalman)

mse_red = np.mean(e_red**2)
mse_kalman = np.mean(e_kalman**2)

print(f"Reduced-order observer MSE: {mse_red:.6f}")
print(f"Kalman filter MSE: {mse_kalman:.6f}")

"""### Comparative Analysis between Luenberger and SMO Observers"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import place_poles

# System matrices
A = np.array([[0, 1],
              [0, -3]])
B = np.array([[0],
              [1]])
C = np.array([[1, 0]])

# Time settings
dt = 0.01
t_eval = np.arange(0, 10, dt)

# Step input
def step_function(t):
    return 1.0 if t >= 1 else 0.0

# Observer parameters
np.random.seed(0)
noise_std = 0.05
disturbance_amplitude = 0.2

# Gains
desired_poles = [-5, -6]
L = place_poles(A.T, C.T, desired_poles).gain_matrix.T

lambda_1 = 30   # SMO gain for x1
lambda_2 = 50   # SMO gain for x2
sat_bound = 0.01  # saturation threshold

# Initial states
x = np.array([[0.0], [0.0]])
xhat_luen = np.array([[0.0], [0.0]])
xhat_smo = np.array([[0.0], [0.0]])

# Storage
x_true = []
x_luen = []
x_smo = []

for t in t_eval:
    u = step_function(t)

    # True system
    dx = A @ x + B * u
    dx[1, 0] += disturbance_amplitude * np.sin(2 * np.pi * 0.5 * t)
    x += dx * dt

    y_true = C @ x
    y_meas = y_true + np.random.normal(0, noise_std, size=(1, 1))

    # Luenberger observer update
    dxhat_luen = A @ xhat_luen + B * u + L @ (y_meas - C @ xhat_luen)
    xhat_luen += dxhat_luen * dt

    # Sliding Mode Observer update
    y_est_smo = C @ xhat_smo
    s = y_meas - y_est_smo
    sat = np.clip(s / sat_bound, -1, 1)  # Saturation

    dxhat_smo = np.array([
        [xhat_smo[1, 0] + lambda_1 * sat[0, 0]],
        [-3 * xhat_smo[1, 0] - 2 * y_meas[0, 0] + u + lambda_2 * sat[0, 0]]
    ])
    xhat_smo += dxhat_smo.reshape(2, 1) * dt

    # Store values
    x_true.append(x.flatten())
    x_luen.append(xhat_luen.flatten())
    x_smo.append(xhat_smo.flatten())

# Convert lists to arrays
x_true = np.array(x_true)
x_luen = np.array(x_luen)
x_smo = np.array(x_smo)

# ======================
# 📈 PLOTS
# ======================
plt.figure(figsize=(12, 8))

# x1 comparison
plt.subplot(2, 1, 1)
plt.plot(t_eval, x_true[:, 0], label='x₁ (true)', linewidth=2)
plt.plot(t_eval, x_luen[:, 0], '--', label='x₁ (Luenberger)', linewidth=2)
plt.plot(t_eval, x_smo[:, 0], ':', label='x₁ (SMO)', linewidth=2)
plt.ylabel('x₁')
plt.title('State x₁: True vs Luenberger vs SMO')
plt.legend()
plt.grid(True)

# x2 comparison
plt.subplot(2, 1, 2)
plt.plot(t_eval, x_true[:, 1], label='x₂ (true)', linewidth=2)
plt.plot(t_eval, x_luen[:, 1], '--', label='x₂ (Luenberger)', linewidth=2)
plt.plot(t_eval, x_smo[:, 1], ':', label='x₂ (SMO)', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('x₂')
plt.title('State x₂: True vs Luenberger vs SMO')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ======================
# 📊 Error Metrics
# ======================
e_luen = x_true - x_luen
e_smo = x_true - x_smo

mse_luen = np.mean(e_luen**2, axis=0)
mse_smo = np.mean(e_smo**2, axis=0)

print(f"Luenberger MSE: x₁ = {mse_luen[0]:.6f}, x₂ = {mse_luen[1]:.6f}")
print(f"SMO MSE:        x₁ = {mse_smo[0]:.6f}, x₂ = {mse_smo[1]:.6f}")

"""###Simulator for Hardware Linear Systems"""

!pip install streamlit

!pip install ipywidgets

!pip install pyserial

import serial
import time

# Setup Serial (adjust COM port and baudrate)
try:
    ser = serial.Serial('COM3', 9600, timeout=1)  # Windows: COM3, Linux: /dev/ttyUSB0
    time.sleep(2)  # Wait for Arduino to reset
except:
    ser = None
    print("Could not connect to serial port. Check device.")

def get_sensor_data():
    if ser is None:
        return np.random.normal(0, 1)  # fallback if no serial
    try:
        line = ser.readline().decode().strip()
        return float(line)  # Assume single float output
    except:
        return None

import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider, Dropdown, Textarea, RadioButtons
from scipy.signal import place_poles
import serial
import time

# Time setup
dt = 0.01
duration = 10
t_eval = np.arange(0, duration, dt)

# Step input
step = lambda t: 1.0 if t >= 1 else 0.0

# Saturation helper
def sat(val, bound):
    return np.clip(val / bound, -1, 1)

# Try to establish serial connection (auto-detect mode)
def try_serial_connection():
    try:
        ser = serial.Serial('COM3', 9600, timeout=1)
        time.sleep(2)
        return ser
    except:
        return None

# Function to read sensor data from serial
def get_sensor_data(ser):
    if ser is None:
        print("[Warning] No serial connection detected.")
        return None
    try:
        line = ser.readline().decode().strip()
        return float(line)
    except:
        return None

# Core simulation function
def simulate_observer(observer, noise_std, disturbance_amp, lambda_1, lambda_2,
                      pole1, pole2, q1, q2, r, matrix_input_mode, A_text, B_text, C_text):

    use_serial = False
    ser = None

    if matrix_input_mode == "Manual":
        try:
            A = np.array(eval(A_text), dtype=float)
            B = np.array(eval(B_text), dtype=float)
            C = np.array(eval(C_text), dtype=float)
        except:
            print("Invalid matrix input. Check formatting.")
            return
    else:
        use_serial = True
        ser = try_serial_connection()
        if ser is None:
            print("[Error] No serial device connected. Please check COM port.")
            return
        A = np.array([[0, 1], [0, -3]], dtype=float)
        B = np.array([[0], [1]], dtype=float)
        C = np.array([[1, 0]], dtype=float)

    # Kalman matrices
    Q = np.array([[q1, 0], [0, q2]])
    R = np.array([[r]])
    P = np.eye(2)
    x_kf = np.zeros((2, 1))

    # Initialization
    x = np.zeros((2, 1))
    xhat = np.zeros((2, 1))
    x_true = []
    x_est = []

    if observer == "Luenberger":
        L = place_poles(A.T, C.T, [pole1, pole2]).gain_matrix.T

    for t in t_eval:
        u = step(t)

        if not use_serial:
            dx = A @ x + B * u
            dx[1, 0] += disturbance_amp * np.sin(2 * np.pi * 0.5 * t)
            x += dx * dt
            y = C @ x
            y_meas = y + np.random.normal(0, noise_std, size=(1, 1))
        else:
            sensor_val = get_sensor_data(ser)
            if sensor_val is None:
                continue  # skip this step if bad data
            y_meas = np.array([[sensor_val]])
            y = y_meas  # no ground truth, so y = y_meas

        # Observer logic
        if observer == "Luenberger":
            dxhat = A @ xhat + B * u + L @ (y_meas - C @ xhat)

        elif observer == "SMO":
            s = y_meas - C @ xhat
            dxhat = np.array([
                [xhat[1, 0] + lambda_1 * sat(s[0, 0], 0.01)],
                [-3 * xhat[1, 0] - 2 * y_meas[0, 0] + u + lambda_2 * sat(s[0, 0], 0.01)]
            ]).reshape(2, 1)

        elif observer == "Kalman":
            x_kf = x_kf + (A @ x_kf + B * u) * dt
            P = A @ P @ A.T + Q
            S = C @ P @ C.T + R
            K = P @ C.T @ np.linalg.inv(S)
            x_kf = x_kf + K @ (y_meas - C @ x_kf)
            P = (np.eye(2) - K @ C) @ P
            xhat = x_kf.copy()

        elif observer == "High-Gain":
            gain = np.array([[lambda_1], [lambda_2]])
            dxhat = A @ xhat + B * u + gain @ (y_meas - C @ xhat)

        elif observer == "Reduced-Order":
            x1 = y_meas[0, 0]
            x2_hat = xhat[1, 0]
            dx2_hat = -3 * x2_hat - 2 * x1 + u
            xhat = np.array([[x1], [x2_hat + dx2_hat * dt]])

        elif observer == "Deadbeat":
            L = place_poles(A.T, C.T, [-8, -9]).gain_matrix.T
            dxhat = A @ xhat + B * u + L @ (y_meas - C @ xhat)

        elif observer == "Finite-Time":
            error = y_meas - C @ xhat
            dxhat = A @ xhat + B * u + 100 * np.sign(error)

        if observer != "Kalman":
            xhat += dxhat * dt

        if not use_serial:
            x_true.append(x.flatten())
        else:
            x_true.append([np.nan, np.nan])  # no ground truth

        x_est.append(xhat.flatten())

    x_true = np.array(x_true)
    x_est = np.array(x_est)
    mse = np.nanmean((x_true - x_est) ** 2, axis=0)

    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(t_eval, x_est[:, 0], label="x1 est")
    if not use_serial:
        ax[0].plot(t_eval, x_true[:, 0], label="x1 true")
    ax[0].legend(); ax[0].grid(); ax[0].set_ylabel("x₁")

    ax[1].plot(t_eval, x_est[:, 1], label="x2 est")
    if not use_serial:
        ax[1].plot(t_eval, x_true[:, 1], label="x2 true")
    ax[1].legend(); ax[1].grid(); ax[1].set_ylabel("x₂"); ax[1].set_xlabel("Time (s)")

    mse_str = f"MSE: x1={mse[0]:.4f}, x2={mse[1]:.4f}" if not use_serial else "(Live input mode)"
    plt.suptitle(f"{observer} Observer | {mse_str}")
    plt.tight_layout()
    plt.show()

# --- UI Widgets ---
interact(
    simulate_observer,
    observer=Dropdown(options=["Luenberger", "Kalman", "SMO", "High-Gain", "Reduced-Order", "Deadbeat", "Finite-Time"], value="Luenberger"),
    noise_std=FloatSlider(min=0.0, max=0.1, step=0.01, value=0.05),
    disturbance_amp=FloatSlider(min=0.0, max=1.0, step=0.05, value=0.2),
    lambda_1=FloatSlider(min=0, max=100, step=5, value=30),
    lambda_2=FloatSlider(min=0, max=100, step=5, value=50),
    pole1=FloatSlider(min=-20, max=-0.1, step=0.1, value=-5),
    pole2=FloatSlider(min=-20, max=-0.1, step=0.1, value=-6),
    q1=FloatSlider(min=1e-6, max=1e-2, step=1e-6, value=1e-5, readout_format=".0e"),
    q2=FloatSlider(min=1e-6, max=1e-2, step=1e-6, value=1e-4, readout_format=".0e"),
    r=FloatSlider(min=1e-6, max=1e-2, step=1e-6, value=2.5e-3, readout_format=".0e"),
    matrix_input_mode=RadioButtons(options=["Manual", "Auto-detect "], value="Manual", description="A,B,C Input:"),
    A_text=Textarea(value="[[0, 1], [0, -3]]", layout={'width': '300px'}),
    B_text=Textarea(value="[[0], [1]]", layout={'width': '300px'}),
    C_text=Textarea(value="[[1, 0]]", layout={'width': '300px'})
)
