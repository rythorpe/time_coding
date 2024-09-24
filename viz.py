"""Visualization functions for model simulations."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_inputs_outputs(inputs, outputs, times,
                        rec_traj=None):
    n_trials, n_times, n_inputs = inputs.shape
    _, _, n_outputs = outputs.shape

    n_plots = 2
    if rec_traj is not None:
        n_plots += n_trials

    colors = plt.cm.binary(np.linspace(0.2, 1, n_trials))
    # colors = plt.cm.viridis(np.linspace(0, 1, n_trials))
    fig, axes = plt.subplots(n_plots, 1, sharex=True, figsize=(6, 6))

    for trial_idx, color in zip(range(n_trials), colors):
        # inputs
        axes[0].plot(times, inputs[trial_idx, :, :], c=color, lw=2)

        # recurrent unit trajectories for each trial
        if rec_traj is not None:
            axes[trial_idx + 1].plot(times, rec_traj[trial_idx, :, :],
                                     c=color)
            axes[trial_idx + 1].set_yticks([-1, 0, 1])
            axes[trial_idx + 1].set_ylabel('h(t)')

        # outputs
        axes[-1].plot(times, outputs[trial_idx, :, :], c=color, lw=2)

    axes[0].set_yticks([-2, 0, 2])
    axes[0].set_ylabel('input')

    axes[-1].set_xticks(np.arange(0, np.round(times[-1] + 0.2), 0.2))
    axes[-1].set_xlabel('time (s)')
    axes[-1].set_yticks([0, 1])
    axes[-1].set_ylabel('output')

    return fig