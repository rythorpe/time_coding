"""Visualization functions for model simulations."""

from cycler import cycler

import numpy as np
import torch

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

from utils import gaussian


# set plotting params
custom_params = {'axes.spines.right': False, 'axes.spines.top': False}
sns.set_theme(style='ticks', rc=custom_params)


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


def plot_divergence(divergence, delay_times, perturb_mags, ax):
    colors = plt.cm.inferno_r(np.linspace(0.1, 1, len(perturb_mags)))
    for perturb_mag_idx, perturb_divergence in enumerate(divergence):
        perturb_mag_str = f'{perturb_mags[perturb_mag_idx]}'
        ax.plot(delay_times, perturb_divergence, label=perturb_mag_str,
                c=colors[perturb_mag_idx], lw=2)
    ax.legend()
    # ax.set_yticks([0.5, 1])
    ax.set_xticks([0, 0.5, 1])
    ax.set_ylabel('MSE')
    ax.set_xlabel('time (s)')

    fig = ax.get_figure()
    fig.tight_layout()

    return fig


def plot_learning(losses, max_iter=None):
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    iter_idxs = np.arange(len(losses))
    ax.semilogy(iter_idxs, losses, 'k')
    ax.grid(axis='y')
    ax.grid(which="minor", color="0.9")
    if max_iter is None:
        ub_xtick = iter_idxs[-1]
    else:
        ub_xtick = max_iter
    ax.set_xticks([0, ub_xtick])
    ax.set_xlabel('iteration')
    ax.set_ylabel('loss')
    fig.tight_layout()

    return fig


def plot_state_traj(ext_in, h_units, syn_eff, outputs, targets, times):
    # NB: assumes a single batch/trial
    n_times, n_hidden = h_units.shape
    n_outputs = outputs.shape[1]

    time_mask = times > 0
    times_after_zero = times[time_mask]

    n_hidden_plot = 5  # number of hidden units to plot
    if n_hidden < 5:
        n_hidden_plot = n_hidden

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(6, 6))

    # create colormaps
    cm_hidden = sns.color_palette('colorblind')
    cm_output = plt.cm.viridis_r(np.linspace(0, 1, n_outputs))

    # injected current
    axes[0].set_prop_cycle(cycler('color', cm_hidden))
    axes[0].plot(times, ext_in[:, :n_hidden_plot])
    axes[0].add_patch(Rectangle([-0.05, 0], 0.05, 1.0, ec='none', fc='k',
                                alpha=0.2, zorder=100))
    axes[0].set_ylabel('injected\ncurrent (a.u.)')
    # axes[0].set_yticks([0, 1])

    # recurrent unit trajectories
    axes[1].set_prop_cycle(cycler('color', cm_hidden))
    axes[1].plot(times, h_units[:, :n_hidden_plot])
    axes[1].add_patch(Rectangle([-0.05, 0], 0.05, 1.0, ec='none', fc='k',
                                alpha=0.2, zorder=100))
    axes[1].set_ylabel('normalized\nfiring rate (a.u.)')
    axes[1].set_yticks([0, 1])

    # synaptic utilization (from STP)
    axes[2].set_prop_cycle(cycler('color', cm_hidden))
    axes[2].plot(times, syn_eff[:, :n_hidden_plot])
    axes[2].add_patch(Rectangle([-0.05, 0], 0.05, 1.0, ec='none', fc='k',
                                alpha=0.2, zorder=100))
    axes[2].set_ylabel('synaptic\nefficacy')
    axes[2].set_yticks([0, 1])

    # outputs
    axes[3].set_prop_cycle(cycler('color', cm_output))
    axes[3].plot(times_after_zero, targets[time_mask, :], lw=2, ls=':')
    axes[3].plot(times, outputs, lw=2)
    axes[3].add_patch(Rectangle([-0.05, -1], 0.05, 2.0, ec='none', fc='k',
                                alpha=0.2, zorder=100))
    axes[3].set_xticks(np.arange(0, 1.2, 0.2))
    axes[3].set_xlabel('time (s)')
    axes[3].set_ylabel('normalized\nfiring rate (a.u.)')
    axes[3].set_yticks([-1, 0, 1])

    fig.tight_layout()

    return fig


def plot_all_units(h_units, syn_eff, outputs, targets, times):
    # NB: assumes a single batch/trial
    n_times, n_hidden = h_units.shape
    n_outputs = outputs.shape[1]

    time_mask = times > 0
    times_after_zero = times[time_mask]

    fig, axes = plt.subplots(1, 3, sharex=True, figsize=(8, 2.5))

    # recurrent unit trajectories
    cm_hidden = sns.color_palette('colorblind')
    hid_res_map = axes[0].pcolormesh(times, range(1, n_hidden + 1),
                                     h_units.T, cmap='RdGy',
                                     vmin=-1, vmax=1)
    axes[0].set_title('hidden unit\nresponses')
    axes[0].set_ylabel('unit #')
    axes[0].set_yticks([1, n_hidden])
    cbar_0 = fig.colorbar(hid_res_map, ax=axes[0], ticks=[-1, 0, 1])

    # synaptic efficacy (from STP)
    syn_eff_map = axes[1].pcolormesh(times, range(1, n_hidden + 1),
                                     syn_eff.T, cmap='Greys',
                                     vmin=0, vmax=1)
    axes[1].set_title('synaptic\nefficacy')
    axes[1].set_yticks([1, n_hidden])
    axes[1].set_xlabel('time (s)')
    cbar_1 = fig.colorbar(syn_eff_map, ax=axes[1], ticks=[0, 1])

    # outputs
    colors_output = plt.cm.viridis_r(np.linspace(0, 1, n_outputs))
    out_res_map = axes[2].pcolormesh(times_after_zero,
                                     range(1, n_outputs + 1),
                                     outputs[time_mask, :].T, cmap='RdGy',
                                     vmin=-1, vmax=1)
    peak_idxs = targets.argmax(dim=0)
    peak_times = times[peak_idxs]
    axes[2].scatter(peak_times, range(1, n_outputs + 1), marker='|',
                    c=colors_output, s=80, linewidths=3)
    axes[2].set_title('output unit\nresponses')
    axes[2].set_yticks([1, n_outputs])
    axes[2].set_xticks([0, 1])
    # axes[2].set_ylabel('normalized\nfiring rate (a.u.)')
    cbar_2 = fig.colorbar(out_res_map, ax=axes[2], ticks=[-1, 0, 1])

    fig.tight_layout()

    return fig


def plot_weight_distr(W_hh, W_hh_mask, syn_eff, eff_gain, true_gain, prob_c):
    fig, axes = plt.subplots(1, 1, figsize=(3, 3))
    n_times, n_hidden = syn_eff.shape
    n_weights = W_hh_mask.sum()
    postsyn_weights = true_gain * W_hh[W_hh_mask == 1]
    effective_weights = torch.stack([(syn_eff[t_idx, :] * true_gain * W_hh * W_hh_mask)[W_hh_mask == 1] for t_idx in range(n_times)])
    effective_weights_avg = effective_weights.mean(dim=0)
    # print(effective_weights_avg)
    weight_bins = np.linspace(-4, 4, 30)
    # normalize to produce PMF
    weights = torch.ones_like(postsyn_weights) / n_weights
    axes.hist(postsyn_weights, bins=weight_bins, weights=weights, alpha=0.7,
              label='static')
    pdf, _, _ = axes.hist(effective_weights_avg, bins=weight_bins,
                          weights=weights, alpha=0.7,
                          label='effective (simulated)')

    w_hidden_std = eff_gain / np.sqrt(prob_c * n_hidden)
    theoretic_distr = gaussian(weight_bins, 0, w_hidden_std)
    theoretic_distr /= np.sum(theoretic_distr)  # normalize

    axes.plot(weight_bins, theoretic_distr, 'k:', lw=2,
              label=f'effective (theoretical)')
    axes.set_xlabel('synaptic weight')
    axes.set_ylabel('probability')

    axes.legend()

    fig.tight_layout()

    return fig
