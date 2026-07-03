# LTI Observer Comparison

Simulation and comparison of state estimation techniques for a 2nd-order linear time-invariant (LTI) system under noisy, disturbed conditions. Covers full-order and reduced-order observer design, Luenberger pole placement, Kalman filtering, and sliding-mode observation, with an interactive dashboard for live parameter tuning.

## System Model

State-space representation:

```
ẋ = Ax + Bu
y = Cx + Du
```

Two system variants are used across experiments:

- **Underdamped 2nd-order system**: `A = [[0, 1], [-2, -3]]`, `B = [[0], [1]]`, `C = [[1, 0]]`
- **Integrator-damped system**: `A = [[0, 1], [0, -3]]`, `B = [[0], [1]]`, `C = [[1, 0]]`

Input: unit step function (with configurable activation time). Measurements are corrupted with Gaussian noise (`σ = 0.05` by default), and select experiments inject a sinusoidal disturbance into the second state's dynamics to test observer robustness.

## Observers Implemented

| Observer | Description |
|---|---|
| **Luenberger (full-order)** | Pole placement via `scipy.signal.place_poles`, poles at `[-5, -6]` |
| **Reduced-order** | Estimates only the unmeasured state (`x₂`) directly from `x₁` measurements |
| **Kalman Filter** | Discrete-time KF (full-order and reduced-order variants), tunable `Q`/`R` |
| **Sliding Mode Observer (SMO)** | Saturation-based switching observer, tunable gains `λ₁`, `λ₂` |
| **High-Gain** | Fixed high-gain injection observer |
| **Deadbeat** | Fast-pole Luenberger variant (`[-8, -9]`) |
| **Finite-Time** | Sign-based error injection for finite-time convergence |

## Results Summary

Each experiment reports mean squared error (MSE) per state to compare estimator accuracy under identical noise/disturbance conditions. Key findings (see `results/mse_summary.csv` after running experiments):

- Kalman filtering consistently achieves the lowest MSE under Gaussian measurement noise, as expected given it's the optimal linear estimator for this noise model.
- The reduced-order observer converges faster in clean conditions but is more sensitive to disturbances injected directly into the unmeasured state.
- SMO shows competitive robustness under bounded disturbances but exhibits chattering near the sliding surface, visible in the `x₂` error plots.

## Project Structure

```
LTI-Observer-Comparison/
├── src/
│   ├── system.py            # System matrices, step input, ODE definitions
│   ├── simulate.py          # Shared simulation loop
│   ├── observers/           # One module per observer type
│   └── utils/                # Metrics and plotting helpers
├── notebooks/
│   └── observer_analysis.py  # Original exploratory notebook
├── experiments/               # Numbered, runnable comparison scripts
├── app/
│   ├── streamlit_app.py      # Interactive dashboard
│   └── serial_interface.py   # Optional hardware-in-loop mode (Arduino/serial sensor input)
├── Results/ 
```

## Getting Started

### Installation

```bash
git clone https://github.com/saurav3103/LTI-Observer-Comparison.git
cd LTI-Observer-Comparison
pip install -r requirements.txt
```

### Running an Experiment

```bash
python experiments/02_luenberger_vs_kalman.py
```

### Launching the Interactive Dashboard

```bash
streamlit run app/streamlit_app.py
```

### Hardware-in-the-Loop Mode

To feed live sensor data (e.g. from an Arduino) instead of a simulated system, connect a serial device and select "Auto-detect" mode in the dashboard. Default port is `COM3` (Windows) — update `app/serial_interface.py` for Linux (`/dev/ttyUSB0`) or macOS (`/dev/tty.usbserial-*`).

## Notes

- All observer simulations use fixed-step Euler integration (`dt = 0.01s`) for consistency across methods; this is adequate for the pole locations used here but should be tightened if extending to faster dynamics.
- Random seeds are fixed (`np.random.seed(0)` / `42`) in comparison scripts for reproducibility across observer types.
- The interactive dashboard falls back to simulated Gaussian noise if no serial device is detected.

## License

MIT
