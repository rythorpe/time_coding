"""Main development script for project."""

from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.signal import periodogram
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed

import torch
from torch import nn

from utils import gaussian, get_gaussian_targets
from models import RNN
from train import test_and_get_stats, pre_train, train_force, train_bptt
from viz import plot_stability


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
# n_samps = 3
n_nets_per_samp = 5
# params = {'stp_heterogeneity': ['none', 'homo', 'hetero'],
#           'n_outputs': np.linspace(5, 25, n_samp),
#           'targ_std': np.linspace(0.005, 0.025, n_samp)}
# xx, yy = np.meshgrid(params['n_outputs'], params['targ_std'])
# param_vals = [pt for pt in zip(xx.flatten(), yy.flatten())]
# # repeat samples to get multiple random nets per configuration
# param_vals = np.tile(param_vals, (n_nets_per_samp, 1))
# n_total_trials = param_vals.shape[0]
params = {'perturbation_mag': np.array([1.0, 1.5, 2.0])}


def train_test_random_net(params=None, plot_sim=False):
    '''Call this func on each parallel process.'''
    if params is not None:
        # param_1, param_2 = params
        pass
    metrics = dict()

    # instantiate model, loss function, and optimizer
    n_inputs, n_hidden, n_outputs = 1, 300, 10
    model = RNN(n_inputs=n_inputs, n_hidden=n_hidden,
                n_outputs=n_outputs, echo_state=False)
    model.to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

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
    # r_0 = None
    u_0 = u_0.to(device)
    # u_0 = None

    # plot model output before training
    hidden_sr, output_sr, stats_0 = test_and_get_stats(inputs, targets, times,
                                                       model, loss_fn, h_0,
                                                       r_0, u_0, plot=plot_sim)

    # pre-train
    # max_iter_pretrain = 10
    # for iter_idx in range(max_iter_pretrain):
    #     _ = pre_train(inputs, times, model, h_0)

    # train model weights
    max_iter = 100
    # convergence_reached = False
    loss_per_iter = list()
    for iter_idx in range(max_iter):
        print(f"Iteration {iter_idx + 1}")
        loss, param_dist = train_bptt(inputs, targets, times, model, loss_fn,
                                      optimizer, h_0, r_0, u_0)
        loss_per_iter.append(loss)
        # recent_loss_std = np.std(loss_per_iter[-20:])
        # loss_range = np.max(loss_per_iter) - np.min(loss_per_iter)
        # if recent_loss_std < 1e-3 * loss_range:
        #     convergence_reached = True
        #     break
    print(f"Trial training complete!!")
    # if not convergence_reached:
    #     print(f"Warning: didn't converge (param_dist={param_dist})!!")

    # plot loss across training
    if plot_sim:
        plt.figure()
        plt.plot(loss_per_iter)
        plt.xlabel('iteration')
        plt.ylabel('loss')

    # investigate fitted model
    # plot model output after training
    hidden_sr, output_sr, stats_1 = test_and_get_stats(inputs, targets, times,
                                                       model, loss_fn, h_0,
                                                       r_0, u_0,
                                                       plot=plot_sim)

    # temporal stability: MSE as a function of latency with t<0 perturbations
    n_tests_per_net = 1
    perturb_dur = 0.05  # 50 ms
    perturb_win_mask = np.logical_and(times > -perturb_dur, times < 0)
    times_after_zero = times[times > 0]
    n_perturb = len(params['perturbation_mag'])
    # loss_vs_perturb = np.zeros([n_tests_per_net, n_perturb, n_outputs])
    loss_vs_perturb = np.zeros([n_tests_per_net, n_perturb, len(times_after_zero)])
    for test_idx in range(n_tests_per_net):
        for perturb_idx, perturb_mag in enumerate(params['perturbation_mag']):
            # now, set perturbation magnitude of input before t=0s
            inputs = torch.zeros((n_batches, n_times, n_inputs))
            inputs[:, perturb_win_mask, :] = perturb_mag

            loss_fn_itemized = nn.MSELoss(reduction='none')
            _, _, stats = test_and_get_stats(inputs, targets, times, model,
                                             loss_fn_itemized, h_0, r_0, u_0,
                                             plot=plot_sim)
            loss = stats['loss']
            # weight loss by Gaussian target over time
            # loss = loss * targets[:, times > 0, :]
            loss_vs_perturb[test_idx, perturb_idx, :] = loss.mean(dim=(0, 2))
    loss_vs_perturb = loss_vs_perturb.mean(axis=0)  # avg over rand input conns
    divergence = (loss_vs_perturb /
                  np.tile(loss_vs_perturb[0, :], [n_perturb, 1]))
    metrics['divergence'] = divergence
    metrics['delay_times'] = delay_times
    metrics['response_times'] = times_after_zero

    ####################################################
    return metrics


# run single trial
# res = train_test_random_net(params, plot_sim=True)

# run sweep sequentially
# for param in param_vals:
#     train_test_random_net(params)

# run sweep in parallel
res = Parallel(n_jobs=10)(delayed(train_test_random_net)(params)
                          for idx in range(n_nets_per_samp))

metrics = defaultdict(list)
for key in res[0].keys():
    for trial in res:
        metrics[key].append(trial[key])

# stability = res['stability']
# delay_times = res['delay_times']
divergence = np.mean(metrics['divergence'], axis=0)
# delay_times = metrics['delay_times'][0]
delay_times = metrics['response_times'][0]
perturb_mags = params['perturbation_mag']
fig_stability = plot_stability(divergence, delay_times, perturb_mags)
