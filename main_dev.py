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

from utils import gaussian, get_gaussian_targets, get_random_targets
from models import RNN
from train import test, pre_train, train, set_optimimal_w_out
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
# n_samp = 3
n_nets_per_samp = 30
# params = {'n_outputs': np.linspace(5, 25, n_samp),
#           'targ_std': np.linspace(0.005, 0.025, n_samp)}
# xx, yy = np.meshgrid(params['n_outputs'], params['targ_std'])
# param_vals = [pt for pt in zip(xx.flatten(), yy.flatten())]
# # repeat samples to get multiple random nets per configuration
# param_vals = np.tile(param_vals, (n_nets_per_samp, 1))
# n_total_trials = param_vals.shape[0]
params = {'perturbation_mag': np.array([0.0, 0.005, 0.015, 0.020, 0.025])}


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
    # print(model)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # set parameters
    # simulation parameters
    dt = 1e-3  # 1 ms
    tstop = 1.1  # 1 sec
    times = np.arange(-0.1, tstop, dt)
    n_times = len(times)

    # define inputs (for contextual modulation / recurrent perturbations)
    n_batches = 1
    inputs = torch.zeros((n_batches, n_times, n_inputs))

    # define output targets
    targ_std = 0.03  # 30 ms
    targets, delay_times_ = get_gaussian_targets(n_batches, n_outputs, times,
                                                 targ_std)
    # targets, opt_basis = get_random_targets(RNN, inputs,
    #                                         (n_inputs, n_hidden, n_outputs),
    #                                         times, n_opt_basis=n_outputs,
    #                                         plot=True)

    # set initial conditions of recurrent units fixed across iterations of
    # training and testing
    h_0 = (torch.rand(n_hidden) * 2) - 1  # uniform in (-1, 1)
    h_0 = torch.tile(h_0, (n_batches, 1))  # replicate for each batch

    # run opt routine
    # move to desired device
    inputs = inputs.to(device)
    targets = targets.to(device)
    h_0 = h_0.to(device)

    # plot model output before training
    _, _ = test(inputs, targets, times, model, loss_fn, h_0, plot=plot_sim)

    # train model weights
    max_iter = 400
    convergence_reached = False
    loss_per_iter = list()
    for iter_idx in range(max_iter):
        print(f"Iteration {iter_idx + 1}")
        loss, param_dist = train(inputs, targets, times, model, loss_fn,
                                 optimizer, h_0)
        # param_dist = pre_train(inputs, times, model, h_0)
        # loss = param_dist
        loss_per_iter.append(loss)
        if param_dist < 2e-4:
            convergence_reached = True
            break
    # print(f"Trial {sample_idx} training complete!!")
    if not convergence_reached:
        print(f"Warning: didn't converge (param_dist={param_dist})!!")

    if plot_sim:
        plt.figure()
        plt.plot(loss_per_iter)
        plt.xlabel('iteration')
        plt.ylabel('loss')

    # investigate fitted model
    # plot model output after training
    h_t, loss = test(inputs, targets, times, model, loss_fn, h_0,
                     plot=plot_sim)
    h_t_batch = h_t.cpu().squeeze()

    # calculate metrics-of-interest for fitted model sim
    ####################################################
    # avg recurrent cross-correlation
    xcorr = np.corrcoef(h_t_batch, rowvar=False)
    avg_xcorr = xcorr.mean()
    metrics['avg_xcorr'] = avg_xcorr

    # spectral overlap
    fs = 1 / dt
    freqs_, targ_spec = periodogram(gaussian(times[times > 0],
                                             tstop / 2, targ_std), fs=fs)
    hidden_specs = list()
    for h_ts in h_t_batch.T:
        freqs_, hidden_spec = periodogram(h_ts[times > 0], fs=fs)
        hidden_specs.append(hidden_spec)
    hidden_spec = np.mean(hidden_specs, axis=0)
    # normalize spectral densities
    targ_spec /= np.sum(targ_spec)
    hidden_spec /= np.sum(hidden_spec)
    spec_overlap = np.sum(np.min([targ_spec, hidden_spec], axis=0))
    metrics['spec_overlap'] = spec_overlap
    # n_convergence_iters
    n_iters = iter_idx
    metrics['n_iters'] = n_iters
    metrics['convergence'] = int(convergence_reached)

    # final MSE
    metrics['final_mse'] = float(loss)

    # temporal stability: MSE as a function of latency with t<0 perturbations
    n_tests_per_net = 10
    perturb_dur = 0.05  # 50 ms
    perturb_win_mask = np.logical_and(times > -perturb_dur, times < 0)
    n_perturb = len(params['perturbation_mag'])
    loss_vs_perturb = np.zeros([n_tests_per_net, n_perturb, n_outputs])
    for test_idx in range(n_tests_per_net):
        # initialize random conn for I(t)->recurrent units
        w_input_std = 1 / np.sqrt(n_hidden)
        torch.nn.init.normal_(model.W_ih, mean=0.0, std=w_input_std)
        for perturb_idx, perturb_mag in enumerate(params['perturbation_mag']):
            # now, set perturbation magnitude of input before t=0s
            inputs = torch.zeros((n_batches, n_times, n_inputs))
            inputs[:, perturb_win_mask, :] = perturb_mag

            loss_fn_itemized = nn.MSELoss(reduction='none')
            _, loss = test(inputs, targets, times, model, loss_fn_itemized,
                           h_0, plot=False)
            # weight loss by Gaussian target over time
            loss = loss * targets[:, times > 0, :]
            loss_vs_perturb[test_idx, perturb_idx, :] = loss.mean(dim=[0, 1])
    loss_vs_perturb = loss_vs_perturb.mean(axis=0)  # avg over rand input conns
    stability = (np.tile(loss_vs_perturb[0, :], [n_perturb, 1]) /
                 loss_vs_perturb)
    metrics['stability'] = stability
    metrics['delay_times'] = delay_times_

    # solve for optimal model output weights given hidden unit responses
    outputs = set_optimimal_w_out(inputs, targets, times, model, loss_fn,
                                  h_0=h_0, plot=plot_sim)
    # baseline noise
    outputs_batch = outputs.cpu().squeeze()
    output_baseline_noise = outputs_batch[times <= 0, :].std()
    metrics['output_baseline_noise'] = float(output_baseline_noise)
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

stability = np.mean(metrics['stability'], axis=0)
delay_times = metrics['delay_times'][0]
perturb_mags = params['perturbation_mag']
fig_stability = plot_stability(stability, delay_times, perturb_mags)
