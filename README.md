# 🧠 Comparative Analysis of State Observers for Noisy Linear Systems with Hardware Interfacing Support

## 🔍 Overview

This project provides an interactive platform to simulate and analyze **seven different state observers** for noisy linear systems. It allows both **manual input of system matrices (A, B, C)** and **real-time data acquisition** from hardware like Arduino. The observers are evaluated based on estimation accuracy and robustness in the presence of noise and disturbances.

## 🚀 Features

* ✅ Simulation of **7 observer types**:

  * Luenberger Observer
  * Kalman Filter
  * Sliding Mode Observer (SMO)
  * High-Gain Observer
  * Reduced-Order Observer
  * Deadbeat Observer
  * Finite-Time Observer
* 📡 **Hardware integration** via Serial (Arduino)
* 📊 Real-time **visualization** of true vs estimated states
* 📉 Estimation **error analysis and Mean Squared Error (MSE)** comparison
* 🛠️ Manual or auto-detected system model inputs

## 🛠️ Requirements

Run on **Google Colab** with the following packages (auto-installed):

```bash
pip install numpy matplotlib scipy ipywidgets pyserial streamlit
```

## ⚙️ Usage

### 1. Manual Simulation

* Select observer type
* Enter system matrices A, B, C
* Adjust noise, disturbance, pole placements, and observer gains
* Run simulation and visualize estimation performance

### 2. Hardware (Arduino) Integration

* Connect to `COM3` or change port in code
* Set input mode to `Auto-detect`
* Observe state estimation in real time from sensor input

## 📁 File Structure

| File                                                                                                    | Description                       |
| ------------------------------------------------------------------------------------------------------- | --------------------------------- |
| `comparative_analysis_of_state_observers_for_noisy_linear_systems_with_hardware_interfacing_support.py` | Full Colab-compatible script      |
| `README.md`                                                                                             | Project documentation (this file) |

## 🔒 License & Code Sharing

You are free to run and explore this project for educational purposes. However, **reproduction or reuse of the code is restricted**.

> **To share without giving away code**, consider:
>
> * Sharing **only screenshots** of the results.
> * Deploying the project as a **web app (e.g., with Streamlit)** and sharing a public link.
> * Making the GitHub repository **private** and giving access only to reviewers.

## 🧪Outputs
<img width="1189" height="590" alt="image" src="https://github.com/user-attachments/assets/7ba23b8e-54ff-4457-ad8b-57b94029ba55" />

<img width="1189" height="590" alt="image" src="https://github.com/user-attachments/assets/629e0970-0746-406a-9b2a-d3aecbdc6eca" />

<img width="1189" height="590" alt="image" src="https://github.com/user-attachments/assets/7e84f12e-4ff1-42ee-b33c-13156b87bba4" />

<img width="989" height="592" alt="image" src="https://github.com/user-attachments/assets/608b44a9-08bf-4d32-abf2-8ac46de18f83" />


## 🧠 Future Scope

* Extend to **non-linear systems** using EKF/UKF
* Deploy as a **dashboard or embedded GUI**
* Add support for **multi-input multi-output (MIMO)** systems
* Implement **adaptive and neural observers**

---
