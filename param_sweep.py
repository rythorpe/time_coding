import os.path as op

import h5py
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from scipy import optimize

import torch
from torch import nn

from utils import get_gaussian_targets, get_commit_hash, get_timestamp
from models import RNN
from train import sim_batch

# tau, include_stp, noise_tau, noise_std, include_corr_noise, p_rel_range
sim_params_all = [[0.01, False, 0.01, 0.0, False, 2],
                  [0.01, True, 0.01, 0.0, False, 2],
                  [0.01, False, 0.01, 1e-6, False, 2],
                  [0.01, True, 0.01, 1e-6, False, 2],
                  [0.01, False, 0.01, 1e-6, True, 2],
                  [0.01, True, 0.01, 1e-6, True, 2],
                  [0.01, True, 0.01, 1e-6, False, 1],
                  [0.01, True, 0.01, 1e-6, False, 0]]
n_random_nets = 30
n_jobs = 30
n_trials = 50
output_dir = '/projects/ryth7446/time_coding_output'
# output_dir = '/home/ryan/Desktop'


def adjust_gain_stp(model, times, dt, n_steps=8):
    '''Adjust network gain to compensate for STP in-place.

    Returns initial gain value.
    '''

    # define inputs (for contextual modulation / recurrent perturbations)
    # NB: simulates a single trial between adjustments
    n_inputs = model.n_inputs
    n_hidden = model.n_hidden
    n_outputs = model.n_outputs
    n_times = len(times)
    inputs = torch.zeros((1, n_times, n_inputs))
    perturb_dur = 0.05  # 50 ms
    perturb_win_mask = np.logical_and(times > -perturb_dur, times < 0)
    inputs[:, perturb_win_mask, :] = 0.1
    noise_tau = 0.01
    noise_std = 0.0
    include_corr_noise = False

    # define output targets
    # set std s.t. amplitude decays to 1/e at intersection with next target
    targ_std = 0.05 / np.sqrt(2)  # ~35 ms
    # tile center of target delays spanning sim duration (minus margins)
    delay_times = np.linspace(0.1, 1.0, n_outputs)
    targets = get_gaussian_targets(1, delay_times, times, targ_std)

    # set initial conditions of recurrent units fixed across iterations of
    # training and testing
    h_0 = torch.zeros(n_hidden)  # steady-state for postsyn activity var
    h_0 = torch.tile(h_0, (1, 1))  # replicate for each batch
    r_0 = torch.ones(n_hidden)  # steady-state for depression var
    r_0 = torch.tile(r_0, (1, 1))
    u_0 = model.p_rel.detach()  # steady-state for facilitation var
    u_0 = torch.tile(u_0, (1, 1))

    # run opt routine
    # move to desired device
    inputs = inputs.to(device)
    targets = targets.to(device)
    h_0 = h_0.to(device)
    r_0 = r_0.to(device)
    u_0 = u_0.to(device)

    # move to desired device
    model.to(device)

    # (re)set gain to its baseline value
    model.gain = model._init_gain

    # slowly inclrease STP beta param, turning up gain adjustment accordingly
    beta_steps = torch.linspace(0, model.beta, n_steps + 1)
    gain_vals = list()
    for beta in beta_steps:
        model.beta = beta

        # adjust beta twice for each incremental increase in beta
        for update_idx in range(2):
            state_vars_, _ = sim_batch(inputs, targets, times, model, loss_fn,
                                       h_0, r_0, u_0, dt=dt,
                                       include_stp=True,
                                       noise_tau=noise_tau,
                                       noise_std=noise_std,
                                       include_corr_noise=include_corr_noise)
            ext_in_, hidden_sr_t_, r_t_, u_t_, output_sr_t_ = state_vars_
            syn_eff_ = r_t_ * u_t_
            adjustment_fctr = model.p_rel.mean() / syn_eff_.mean()
            model.gain = model._init_gain * adjustment_fctr
            gain_vals.append(model.gain.item())

    return gain_vals


def sim_net(model, loss_fn, times,
            tau, include_stp, noise_tau, noise_std, include_corr_noise,
            p_rel_range, adjusted_gain,
            n_trials=100, dt=1e-3,
            device='cpu'):
    '''Train current instantiation of network model.

    Returns network parameters and variables for a batch of sim trials.
    '''

    # define inputs
    n_inputs = model.n_inputs
    n_hidden = model.n_hidden
    n_outputs = model.n_outputs
    n_times = len(times)
    inputs = torch.zeros((n_trials, n_times, n_inputs))
    perturb_dur = 0.05  # 50 ms
    perturb_win_mask = np.logical_and(times > -perturb_dur, times < 0)
    inputs[:, perturb_win_mask, :] = 0.1

    model.tau = tau
    if model.tau > 0.01:
        # increase input strength
        inputs[:, perturb_win_mask, :] = 0.3
    if include_stp:
        model.gain = adjusted_gain
    else:
        model.gain = model._init_gain
    if p_rel_range == 2:
        p_rel_range = (0.1, 0.9)  # high heterogeneity
    elif p_rel_range == 1:
        p_rel_range = (0.4, 0.6)  # low heterogeneity
    elif p_rel_range == 0:
        p_rel_range = (0.5, 0.5)  # homogeneous
    torch.nn.init.uniform_(model.p_rel, a=p_rel_range[0], b=p_rel_range[1])

    # define output targets
    # set std s.t. amplitude decays to 1/e at intersection with next target
    targ_std = 0.05 / np.sqrt(2)  # ~35 ms
    # tile center of target delays spanning sim duration (minus margins)
    delay_times = np.linspace(0.1, 1.0, n_outputs)
    targets = get_gaussian_targets(n_trials, delay_times, times, targ_std)

    # set initial conditions of recurrent units fixed across iterations of
    # training and testing
    # h_0 = torch.tensor(sol.x[:n_hidden], dtype=torch.float32)
    h_0 = torch.zeros(n_hidden)
    h_0 = torch.tile(h_0, (n_trials, 1))  # replicate for each batch
    # r_0 = torch.tensor(sol.x[n_hidden:2 * n_hidden], dtype=torch.float32)
    r_0 = torch.ones(n_hidden) 
    r_0 = torch.tile(r_0, (n_trials, 1))
    # u_0 = torch.tensor(sol.x[2 * n_hidden:3 * n_hidden], dtype=torch.float32)
    u_0 = model.p_rel.detach()
    u_0 = torch.tile(u_0, (n_trials, 1))

    # run opt routine
    # move to desired device
    model.to(device)
    inputs = inputs.to(device)
    targets = targets.to(device)
    h_0 = h_0.to(device)
    r_0 = r_0.to(device)
    u_0 = u_0.to(device)
 
    state_vars_raw, _ = sim_batch(
        inputs, targets, times, model, loss_fn, h_0, r_0, u_0, dt,
        include_stp=include_stp, noise_tau=noise_tau,
        noise_std=noise_std, include_corr_noise=include_corr_noise)
    
    ext_in, h_t, r_t, u_t, z_t = state_vars_raw
    state_vars = (ext_in.detach().numpy(),
                  model.transfer_func(h_t).detach().numpy(),
                  r_t.detach().numpy(),
                  u_t.detach().numpy(),
                  z_t.detach().numpy())

    return state_vars


def loss_fn(output, target):
    mse_fn = nn.MSELoss()
    return mse_fn(output, target) / (target ** 2).mean()


def eval_net_instance(sim_params_all, net_idx):
    '''Sweeps over sim params for a given random net, then saves output.'''
    # set simulation parameters
    dt = 1e-3  # 1 ms
    tstop = 1.2  # 1 sec
    times = np.arange(-0.1 + dt, tstop + dt, dt)
    n_times = len(times)

    # create network and scale up gain to accomodate STP
    n_inputs, n_hidden, n_outputs = 1, 500, 10
    sample_new_net = True
    while sample_new_net is True:
        # instantiate network
        model = RNN(n_inputs=n_inputs, n_hidden=n_hidden,
                    n_outputs=n_outputs)

        # run gain adjustment; modifies gain in-place
        print('start: gain adjustment')
        gains = adjust_gain_stp(model, times, n_steps=8, dt=dt)
        print('end: gain adjustment')

        # check for an abberant drop in gain; if not, pass
        gain_diffs = np.diff(gains)
        increases = gain_diffs[gain_diffs > 0]
        decreases = gain_diffs[gain_diffs <= 0]
        if np.abs(decreases).max() < 2 * increases.max():
            sample_new_net = False
        else:
            print('warning: resampling network')

    # save adjusted gain value for later
    adjusted_gain = gains[-1]

    n_sims = len(sim_params_all)
    model_params = {'W_hz': model.W_hz.data.detach().clone(),
                    'offset_hz': model.offset_hz.data.detach().clone()}
    state_vars_all = {'ext_in': np.empty((n_sims, n_trials, n_times,
                                          n_hidden), dtype=np.float32),
                      'h_sp': np.empty((n_sims, n_trials, n_times,
                                       n_hidden), dtype=np.float32),
                      'r': np.empty((n_sims, n_trials, n_times,
                                       n_hidden), dtype=np.float32),
                      'u': np.empty((n_sims, n_trials, n_times,
                                       n_hidden), dtype=np.float32)}

    for sim_idx, sim_params in enumerate(sim_params_all):
        tau, include_stp, noise_tau, noise_std, include_corr_noise, p_rel_range = sim_params

        print(f'start: simulated trials for network condition {sim_idx}')
        state_vars = sim_net(
            model=model,
            loss_fn=loss_fn,
            times=times,
            tau=tau,
            include_stp=include_stp,
            noise_tau=noise_tau,
            noise_std=noise_std,
            include_corr_noise=include_corr_noise,
            p_rel_range=p_rel_range,
            adjusted_gain=adjusted_gain,
            n_trials=n_trials,
            dt=dt,
            device=device)
        print(f'end: simulated trials for network condition {sim_idx}')

        state_vars_all['ext_in'][sim_idx] = state_vars[0]
        state_vars_all['h_sp'][sim_idx] = state_vars[1]
        state_vars_all['r'][sim_idx] = state_vars[2]
        state_vars_all['u'][sim_idx] = state_vars[3]

    # save simulation data in HDF5 file within output directory
    fname_local = ('sim_data_' + f'net{net_idx:02d}_' +
                   get_commit_hash() + '_' + get_timestamp() + '.hdf5')
    fname_absolute = op.join(output_dir, fname_local)
    with h5py.File(fname_absolute, 'w') as f_write:
        f_write.create_dataset('sim_params', data=np.array(sim_params_all))
        for key, val in model_params.items():
            f_write.create_dataset(key, data=val)
        for key, val in state_vars_all.items():
            f_write.create_dataset(key, data=val)


if __name__ == '__main__':
    # set meta-parameters
    # for plotting style
    custom_params = {'axes.spines.right': False, 'axes.spines.top': False}
    sns.set_theme(style='ticks', rc=custom_params)
    # for pytorch tensors
    device = 'cpu'
    # for reproducibility; numpy is for model sparse conns
    torch.random.manual_seed(93214)
    np.random.seed(35107)

    Parallel(n_jobs=n_jobs)(delayed(eval_net_instance)
                            (sim_params_all, net_idx)
                            for net_idx in range(n_random_nets))
