"""
Shared plotting helpers for state trajectories and multi-observer comparisons.
"""

import matplotlib.pyplot as plt


def plot_states(t_eval, x_true, x_hat, title="Observer Performance", labels=None):
    """
    Plot true vs. estimated states, one subplot per state dimension.

    labels: optional list of per-state y-axis labels, e.g. ["x1", "x2"].
    """
    n = x_true.shape[1]
    labels = labels or [f"x{i+1}" for i in range(n)]

    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(t_eval, x_true[:, i], label=f"{labels[i]} (true)", linewidth=2)
        ax.plot(t_eval, x_hat[:, i], '--', label=f"{labels[i]} (estimated)", linewidth=2)
        ax.set_ylabel(labels[i])
        ax.legend()
        ax.grid(True)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title)
    plt.tight_layout()
    return fig


def plot_comparison(t_eval, x_true, estimates, title="Observer Comparison", state_index=0,
                     state_label="x"):
    """
    Overlay the true state against multiple observer estimates for one
    state dimension.

    estimates: dict of {observer_name: x_hat_array} where each x_hat_array
               is (N, n); only `state_index` is plotted from each.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_eval, x_true[:, state_index], label=f"{state_label} (true)", linewidth=2)

    styles = ['--', ':', '-.', (0, (3, 1, 1, 1)), (0, (5, 1))]
    for i, (name, x_hat) in enumerate(estimates.items()):
        style = styles[i % len(styles)]
        ax.plot(t_eval, x_hat[:, state_index], style, label=f"{state_label} ({name})", linewidth=2)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(state_label)
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig
