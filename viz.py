"""Visualization functions for model simulations."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_inputs_outputs(inputs, outputs, times, rec_traj=None, targets=None):
    n_trials, n_times, n_inputs = inputs.shape
    _, _, n_outputs = outputs.shape

    n_plots = 2
    if rec_traj is not None:
        n_plots += n_trials
        n_hidden = rec_traj.shape[2]
        n_hidden_plot = 10  # number of hidden units to plot
        if n_hidden < 10:
            n_hidden_plot = n_hidden

    colors = plt.cm.binary(np.linspace(0.2, 1, n_trials))
    # colors = plt.cm.viridis(np.linspace(0, 1, n_trials))
    fig, axes = plt.subplots(n_plots, 1, sharex=True, sharey=True,
                             figsize=(6, 6))

    for trial_idx, color in zip(range(n_trials), colors):
        # inputs
        axes[0].plot(times, inputs[trial_idx, :, :], c=color, lw=2)

        # recurrent unit trajectories for each trial
        if rec_traj is not None:
            axes[trial_idx + 1].plot(times,
                                     rec_traj[trial_idx, :, :n_hidden_plot],
                                     c=color)
            # axes[trial_idx + 1].set_yticks([-1, 0, 1])
            axes[trial_idx + 1].set_ylabel('X (hidden)')

        # outputs
        axes[-1].plot(times[times > 0], outputs[trial_idx, times > 0, :],
                      c=color, lw=2)
        if targets is not None:
            axes[-1].plot(times, targets[trial_idx, :, :],
                          c=color, lw=2, ls=':')

    # axes[0].set_yticks([-2, 0, 2])
    axes[0].set_ylabel('I')

    axes[-1].set_xticks(np.arange(0, times[-1] + 0.2, 0.2))
    axes[-1].set_xlabel('time (s)')
    # axes[-1].set_yticks([0, 1])
    axes[-1].set_ylabel('z')

    return fig


def plot_traj(h_units, outputs, targets, times):
    # NB: assumes a single batch/trial
    n_times, n_hidden = h_units.shape
    n_outputs = outputs.shape[1]

    time_mask = times > 0
    times_after_zero = times[time_mask]

    n_hidden_plot = 10  # number of hidden units to plot
    if n_hidden < 10:
        n_hidden_plot = n_hidden

    # colors = plt.cm.binary(np.linspace(0.2, 1, n_outputs))
    colors = plt.cm.inferno(np.linspace(0, 1, n_outputs))
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6, 6))

    # recurrent unit trajectories
    axes[0].plot(times, h_units[:, :n_hidden_plot], c='k')
    axes[0].set_ylabel('recurrent\nunits')

    # outputs
    for out_idx, color in zip(range(n_outputs), colors):
        axes[1].plot(times, outputs[:, out_idx], c=color,
                     lw=2)
        axes[1].plot(times_after_zero, targets[time_mask, out_idx], c=color,
                     lw=1, ls=':')

    axes[1].set_xticks(np.arange(0, 1.2, 0.2))
    axes[1].set_xlabel('time (s)')
    axes[1].set_ylabel('output')

    return fig
