"""Main development script for project."""

from collections import defaultdict
import os.path as op

import numpy as np
import pandas as pd
from scipy.signal import periodogram
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed

import torch
from torch import nn, autograd

from utils import gaussian, get_gaussian_targets
from models import RNN
from train import test_and_get_stats, pre_train, train_bptt
from viz import (plot_divergence, plot_learning, plot_state_traj,
                 plot_all_units)


# set meta-parameters
# for plotting style
custom_params = {'axes.spines.right': False, 'axes.spines.top': False}
sns.set_theme(style='ticks', rc=custom_params)
# for pytorch tensors
device = 'cpu'
# for reproducibility while troubleshooting; numpy is for model sparse conns
torch.random.manual_seed(95214)
np.random.seed(35107)
output_dir = '/projects/ryth7446/time_coding_output'


# define parameter sweep
n_nets_per_param = 30
param_labels = ['high-hetero', 'low-hetero', 'homo', 'none']
params = {'stp_heterogeneity': [(0.1, 0.9), (0.4, 0.6), (0.5, 0.5), 'none']}
param_vals = np.tile(np.array(params['stp_heterogeneity'], dtype=object),
                     (n_nets_per_param,))
param_keys = np.tile(np.array(param_labels, dtype=object),
                     (n_nets_per_param,))


def train_test_random_net(param_val, plot_sim=False, net_label=None):
    '''Call this func on each parallel process.'''

    # sim/sweep params for a given network instantiation
    perturbation_mag = np.array([1.0, 1.5, 2.0])
    p_rel_range = (0.1, 0.9)
    if param_val == 'none':
        include_stp = False
    else:
        include_stp = True
        p_rel_range = param_val

    metrics = dict()
    metrics['perturbation_mag'] = perturbation_mag

    # instantiate model, loss function, and optimizer
    n_inputs, n_hidden, n_outputs = 1, 500, 10
    resample_net = True
    while resample_net is True:
        model = RNN(n_inputs=n_inputs, n_hidden=n_hidden,
                    n_outputs=n_outputs, p_rel_range=p_rel_range,
                    include_stp=include_stp)
        model.to(device)

        mse_fn = nn.MSELoss()
        # normalize by loss if network output flatlines
        loss_fn = lambda a, b: mse_fn(a, b) / b.mean()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

        # set parameters
        # simulation parameters
        dt = 1e-3  # 1 ms
        tstop = 1.2  # 1 sec
        times = np.arange(-0.1 + dt, tstop + dt, dt)
        n_times = len(times)

        # define inputs (for contextual modulation / recurrent perturbations)
        n_batches = 1
        inputs = torch.zeros((n_batches, n_times, n_inputs))
        perturb_dur = 0.05  # 50 ms
        perturb_win_mask = np.logical_and(times > -perturb_dur, times < 0)
        inputs[:, perturb_win_mask, :] = 1.0

        # define output targets
        # set std s.t. amplitude decays to 1/e at intersection with next target
        targ_std = 0.05 / np.sqrt(2)  # ~35 ms
        # tile center of target delays spanning sim duration (minus margins)
        delay_times = np.linspace(0.1, 1.0, n_outputs)
        targets = get_gaussian_targets(n_batches, delay_times, times, targ_std)

        # set initial conditions of recurrent units fixed across iterations of
        # training and testing
        h_0 = torch.zeros(n_hidden)  # steady-state for postsyn activity var
        h_0 = torch.tile(h_0, (n_batches, 1))  # replicate for each batch
        r_0 = torch.ones(n_hidden)  # steady-state for depression var
        r_0 = torch.tile(r_0, (n_batches, 1))
        u_0 = model.p_rel.detach()  # steady-state for facilitation var
        u_0 = torch.tile(u_0, (n_batches, 1))

        # run opt routine
        # move to desired device
        inputs = inputs.to(device)
        targets = targets.to(device)
        h_0 = h_0.to(device)
        r_0 = r_0.to(device)
        u_0 = u_0.to(device)

        # plot model output before training
        _, sim_stats_0 = test_and_get_stats(inputs, targets, times,
                                            model, loss_fn,
                                            h_0, r_0, u_0, plot=False)

        # pre-train
        # max_iter_pretrain = 10
        # for iter_idx in range(max_iter_pretrain):
        #     _ = pre_train(inputs, times, model, h_0)

        # train model weights
        max_iter = 400
        convergence_reached = False
        resample_net = False
        loss_per_iter = list()
        for iter_idx in range(max_iter):
            # print(f"Iteration {iter_idx + 1}")
            loss = train_bptt(inputs, targets, times, model, loss_fn,
                              optimizer, h_0, r_0, u_0)

            # escape training and resample network if loss/gradient becomes
            # unstable
            if not np.isfinite(loss):
                resample_net = True
                print('Warning: unstable gradient! Resampling network.')
                break

            # collect loss value and evaluate whether or not convergence
            # has been reached
            loss_per_iter.append(loss)
            if len(loss_per_iter) >= 10:
                mean_diff = np.diff(loss_per_iter[-10:]).mean()
                if np.abs(mean_diff) < 1e-4:
                    convergence_reached = True
                    # print(f'Trial training complete for {net_label}')
                    # break

    if convergence_reached is False:
        print('Warning: convergence not reached!!!')

    # investigate fitted model
    # plot model output after training
    state_vars_1, sim_stats_1 = test_and_get_stats(inputs, targets, times,
                                                   model, loss_fn,
                                                   h_0, r_0, u_0, plot=False)
    hidden_sr_1, r_1, u_1, output_sr_1 = state_vars_1

    loss_per_iter.append(sim_stats_1['loss'].item())
    lr_auc = np.mean(np.array(loss_per_iter) - loss_per_iter[-1])
    half_loss = loss_per_iter[0] - (loss_per_iter[0] + loss_per_iter[-1]) / 2
    lr_halflife_idxs = np.nonzero(np.array(loss_per_iter) < half_loss)[0]
    if len(lr_halflife_idxs) > 0:
        lr_halflife = lr_halflife_idxs[0]
    else:
        lr_halflife = 0

    # plot results of training
    if plot_sim:
        # fig_learning = plot_learning(loss_per_iter)
        # axes = fig_learning.get_axes()
        # axes[0].set_title(f'final loss: {loss_per_iter[-1]:.5f}\n'
        #                   f'LR (AUC):{lr_auc:.5f}\n'
        #                   f'LR (halflife): {lr_halflife}')
        # fig_learning.tight_layout()
        # fname = 'learning_loss_' + net_label + '.pdf'
        # fig_learning.savefig(op.join(output_dir, fname))

        # select first batch if more than one exists
        targets_batch = targets.cpu()[0]

        fig_traj = plot_state_traj(h_units=hidden_sr_1[0],
                                   syn_eff=r_1[0] * u_1[0],
                                   outputs=output_sr_1[0],
                                   targets=targets_batch, times=times)
        fname = 'state_traj_' + net_label + '.pdf'
        fig_traj.savefig(op.join(output_dir, fname))

        fig_all_units = plot_all_units(h_units=hidden_sr_1[0],
                                       syn_eff=r_1[0] * u_1[0],
                                       outputs=output_sr_1[0],
                                       targets=targets_batch, times=times)
        fname = 'all_units_' + net_label + '.pdf'
        fig_all_units.savefig(op.join(output_dir, fname))

    # temporal stability: MSE as a function of latency with t<0 perturbations
    n_tests_per_net = 1
    perturb_dur = 0.05  # 50 ms
    perturb_win_mask = np.logical_and(times > -perturb_dur, times < 0)
    times_mask = np.logical_and(times > 0.0, times <= 1.0)
    times_after_zero = times[times_mask]
    n_perturb = len(perturbation_mag)
    # loss_vs_perturb = np.zeros([n_tests_per_net, n_perturb, n_outputs])
    mse_vs_perturb = np.zeros([n_tests_per_net, n_perturb,
                               len(times_after_zero)])
    mse_fn = nn.MSELoss(reduction='none')
    for test_idx in range(n_tests_per_net):
        for perturb_idx, perturb_mag in enumerate(perturbation_mag):
            # now, set perturbation magnitude of input before t=0s
            inputs = torch.zeros((n_batches, n_times, n_inputs))
            inputs[:, perturb_win_mask, :] = perturb_mag

            state_vars_2, sim_stats_2 = test_and_get_stats(inputs, targets,
                                                           times, model,
                                                           loss_fn,
                                                           h_0, r_0, u_0,
                                                           plot=False)
            _, _, _, output_sr_2 = state_vars_2

            mse = mse_fn(output_sr_2[:, times_mask, :],
                         targets[:, times_mask, :])
            mse_vs_perturb[test_idx, perturb_idx, :] = mse.mean(dim=(0, 2))
    divergence = mse_vs_perturb.mean(axis=0)  # avg over rand input conns

    metrics['divergence'] = divergence
    metrics['response_times'] = times_after_zero
    metrics['losses'] = loss_per_iter
    metrics['lr_auc'] = lr_auc
    metrics['lr_halflife'] = lr_halflife
    metrics['n_learning_trials'] = iter_idx + 1
    metrics['final_dim'] = sim_stats_1['dimensionality']
    metrics['dim_diff'] = (sim_stats_1['dimensionality'] -
                           sim_stats_0['dimensionality'])

    return metrics


# run single trial
# res = train_test_random_net(params, plot_sim=True)

# run sweep sequentially
# for param_val in param_vals:
#     train_test_random_net(param_val, plot_sim=True)

# run sweep in parallel
res = Parallel(n_jobs=24)(delayed(train_test_random_net)
                          (param_val, True,
                           param_keys[param_idx] + f'_{param_idx}')
                          for param_idx, param_val in enumerate(param_vals))

# parse data
learning_metrics = defaultdict(list)
learning_metrics['stp'] = param_keys.tolist()
divergences = list()
response_times = list()
perturbation_mags = list()
stp_types = np.repeat(param_keys, 3)

for key in res[0].keys():
    for trial in res:
        if trial[key] is not None:
            if key == 'divergence':
                divergences.extend(trial[key])
            elif key == 'perturbation_mag':
                perturbation_mags.extend(trial[key])
            elif key == 'response_times':
                response_times.extend(np.tile(trial[key], (3, 1)).tolist())
            else:
                learning_metrics[key].append(trial[key])

# fig_divergence, ax = plt.subplots(1, 1, figsize=(3, 3))
# divergence = np.mean(metrics['divergence'], axis=0)
# delay_times = metrics['response_times'][0]
# perturb_mags = metrics['perturbation_mag'][0]
# plot_divergence(divergence, delay_times, perturb_mags, ax=ax)
# fig_divergence.tight_layout()

n_times = len(response_times[0])
data = np.array([np.ravel(divergences), np.ravel(response_times),
                 np.repeat(perturbation_mags, n_times),
                 np.repeat(stp_types, n_times)])
div_df = pd.DataFrame(data.T, columns=['MSE', 'time (s)', 'perturbation',
                                       'stp_type'])
fig_divergence, axes = plt.subplots(1, len(param_labels), sharey=True,
                                    figsize=(10, 3))
for stp_type_idx, stp_type in enumerate(param_labels):
    sns.lineplot(data=div_df[div_df['stp_type'] == stp_type], x='time (s)',
                 y='MSE', hue='perturbation', ax=axes[stp_type_idx])
fig_divergence.tight_layout()
fname = 'divergence.pdf'
fig_divergence.savefig(op.join(output_dir, fname))
