"""Main development script for project."""

from collections import defaultdict

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
from viz import plot_divergence, plot_learning


# set meta-parameters
# for plotting style
custom_params = {'axes.spines.right': False, 'axes.spines.top': False}
sns.set_theme(style='ticks', rc=custom_params)
# for pytorch tensors
device = 'cpu'
# for reproducibility while troubleshooting; numpy is for model sparse conns
torch.random.manual_seed(95214)
np.random.seed(35107)


# define parameter sweep
n_nets_per_param = 1
params = {'stp_heterogeneity': [(0.1, 0.9), (0.4, 0.6), (0.5, 0.5), 'none']}
param_vals = np.tile(np.array(params['stp_heterogeneity'], dtype=object),
                     (n_nets_per_param,))


def train_test_random_net(param_val, plot_sim=False):
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
    n_inputs, n_hidden, n_outputs = 1, 300, 10
    model = RNN(n_inputs=n_inputs, n_hidden=n_hidden,
                n_outputs=n_outputs, p_rel_range=p_rel_range,
                include_stp=include_stp)
    model.to(device)

    mse_fn = nn.MSELoss()
    # normalize by loss if network output flatlines
    loss_fn = lambda a, b: mse_fn(a, b) / b.mean()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

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
    _, _, stats_0 = test_and_get_stats(inputs, targets, times, model,
                                       loss_fn, h_0, r_0, u_0, plot=plot_sim)

    # pre-train
    # max_iter_pretrain = 10
    # for iter_idx in range(max_iter_pretrain):
    #     _ = pre_train(inputs, times, model, h_0)

    # train model weights
    max_iter = 600
    convergence_reached = False
    loss_per_iter = list()
    for iter_idx in range(max_iter):
        print(f"Iteration {iter_idx + 1}")

        loss = train_bptt(inputs, targets, times, model, loss_fn, optimizer,
                          h_0, r_0, u_0)
        loss_per_iter.append(loss)
        if len(loss_per_iter) >= 10:
            mean_diff = np.diff(loss_per_iter[-10:]).mean()
            if np.abs(mean_diff) < 1e-5:
                convergence_reached = True
                print('Trial training complete!!!')
                break

    if convergence_reached is False:
        print('Warning: convergence not reached!!!')

    # investigate fitted model
    # plot model output after training
    _, output_sr_1, stats_1 = test_and_get_stats(inputs, targets, times,
                                                 model, loss_fn,
                                                 h_0, r_0, u_0,
                                                 plot=plot_sim)
    loss_per_iter.append(stats_1['loss'].item())
    lr_auc = np.sum(np.array(loss_per_iter) - loss_per_iter[-1])
    half_loss = loss_per_iter[0] - (loss_per_iter[0] + loss_per_iter[-1]) / 2
    lr_halflife = np.nonzero(np.array(loss_per_iter) < half_loss)[0][0]

    # plot loss across training
    if plot_sim:
        fig_learning = plot_learning(loss_per_iter)

        axes = fig_learning.get_axes()
        axes[0].set_title(f'finnal loss: {loss_per_iter[-1]}\n'
                          f'LR (AUC):{lr_auc}\n LR (halflife): {lr_halflife}')

    # temporal stability: MSE as a function of latency with t<0 perturbations
    n_tests_per_net = 1
    perturb_dur = 0.05  # 50 ms
    perturb_win_mask = np.logical_and(times > -perturb_dur, times < 0)
    times_mask = np.logical_and(times > 0.0, times <= 1.0)
    times_after_zero = times[times_mask]
    n_perturb = len(perturbation_mag)
    # loss_vs_perturb = np.zeros([n_tests_per_net, n_perturb, n_outputs])
    loss_vs_perturb = np.zeros([n_tests_per_net, n_perturb,
                                len(times_after_zero)])
    mse_fn = nn.MSELoss(reduction='none')
    for test_idx in range(n_tests_per_net):
        for perturb_idx, perturb_mag in enumerate(perturbation_mag):
            # now, set perturbation magnitude of input before t=0s
            inputs = torch.zeros((n_batches, n_times, n_inputs))
            inputs[:, perturb_win_mask, :] = perturb_mag

            _, output_sr_2, stats_2 = test_and_get_stats(inputs, targets,
                                                         times, model,
                                                         loss_fn,
                                                         h_0, r_0, u_0,
                                                         plot=plot_sim)
            mse = mse_fn(output_sr_2[:, times_mask, :],
                         output_sr_1[:, times_mask, :])
            loss_vs_perturb[test_idx, perturb_idx, :] = mse.mean(dim=(0, 2))
    divergence = loss_vs_perturb.mean(axis=0)  # avg over rand input conns

    metrics['divergence'] = divergence
    metrics['response_times'] = times_after_zero
    metrics['final_loss'] = loss_per_iter[-1]
    metrics['lr_auc'] = lr_auc
    metrics['lr_halflife'] = lr_halflife
    metrics['n_learning_trials'] = iter_idx + 1
    metrics['final_dim'] = stats_1['dimensionality']
    metrics['dim_diff'] = stats_1['dimensionality'] - stats_0['dimensionality']

    return metrics


# run single trial
# res = train_test_random_net(params, plot_sim=True)

# run sweep sequentially
for param_val in param_vals:
    train_test_random_net(param_val, plot_sim=True)

# # run sweep in parallel
# res = Parallel(n_jobs=10)(delayed(train_test_random_net)(param_val)
#                           for param_val in param_vals)

# metrics = defaultdict(list)
# for key in res[0].keys():
#     for trial in res:
#         metrics[key].append(trial[key])

# p_rel_labels = ['high-hetero', 'low-hetero', 'homo', 'none']
# divergence = np.mean(metrics['divergence'], axis=0)
# delay_times = metrics['response_times'][0]
# perturb_mags = metrics['perturbation_mag'][0]
# fig_divergence = plot_divergence(divergence, delay_times, perturb_mags)
