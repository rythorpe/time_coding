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

from utils import get_gaussian_targets
from models import RNN
from train import test_and_get_stats, train_bptt
from viz import plot_state_traj, plot_all_units, plot_weight_distr


sim_params_all = [[0.01, False, 0.01, 0.0, False],
                  [0.05, False, 0.01, 0.0, False],
                  [0.01, False, 0.01, 0.0, False],
                  [0.01, True, 0.01, 0.0, False],
                  [0.01, False, 0.01, 1e-6, False],
                  [0.01, True, 0.01, 1e-6, False],
                  [0.01, False, 0.01, 1e-6, True],
                  [0.01, True, 0.01, 1e-6, True]]
n_trials = 500
return_trials = (0, 100, n_trials)
output_dir = '/projects/ryth7446/time_coding_output'


def adjust_gain_stp(model, times, n_steps=8):
    '''Adjust network gain to compensate for STP in-place.

    Returns initial gain value.
    '''

    # define inputs (for contextual modulation / recurrent perturbations)
    n_batches = 1
    inputs = torch.zeros((n_batches, n_times, n_inputs))
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

    # move to desired device
    model.to(device)

    # (re)set gain to its baseline value
    model.gain = model._init_gain

    # slowly inclrease STP beta param, turning up gain adjustment accordingly
    beta_steps = torch.linspace(0, model.beta, n_steps + 1)
    gain_vals = list()
    model.eval()
    for beta in beta_steps:
        model.beta = beta

        # adjust beta twice for each incremental increase in beta
        for update_idx in range(2):
            state_vars_, _ = test_and_get_stats(inputs, targets, times,
                                                model, loss_fn, h_0, r_0, u_0,
                                                include_stp=True,
                                                noise_tau=noise_tau,
                                                noise_std=noise_std,
                                                include_corr_noise=include_corr_noise,
                                                plot=False)
            ext_in_, hidden_sr_t_, r_t_, u_t_, output_sr_t_ = state_vars_
            syn_eff_ = r_t_* u_t_
            adjustment_fctr = model.p_rel.mean() / syn_eff_.mean()
            model.gain = model._init_gain * adjustment_fctr
            gain_vals.append(model.gain.item())

    return gain_vals


def train_net(model, optimizer, loss_fn, times,
              tau, include_stp, noise_tau, noise_std, include_corr_noise,
              adjusted_gain,
              n_trials=1000, return_trials=(0, 100, 1000), device='cpu'):
    '''Train current instantiation of network model.
    
    Returns network parameters and variables after return_trials number of
    training trials.
    '''

    # define inputs
    n_batches = 1
    n_inputs = model.n_inputs
    n_hidden = model.n_hidden
    n_outputs = model.n_outputs
    n_times = len(times)
    inputs = torch.zeros((n_batches, n_times, n_inputs))
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

    # define output targets
    # set std s.t. amplitude decays to 1/e at intersection with next target
    targ_std = 0.05 / np.sqrt(2)  # ~35 ms
    # tile center of target delays spanning sim duration (minus margins)
    delay_times = np.linspace(0.1, 1.0, n_outputs)
    targets = get_gaussian_targets(n_batches, delay_times, times, targ_std)

    # set initial conditions of recurrent units fixed across iterations of
    # training and testing
    # h_0 = torch.tensor(sol.x[:n_hidden], dtype=torch.float32)
    h_0 = torch.zeros(n_hidden)
    h_0 = torch.tile(h_0, (n_batches, 1))  # replicate for each batch
    # r_0 = torch.tensor(sol.x[n_hidden:2 * n_hidden], dtype=torch.float32)
    r_0 = torch.ones(n_hidden) 
    r_0 = torch.tile(r_0, (n_batches, 1))
    # u_0 = torch.tensor(sol.x[2 * n_hidden:3 * n_hidden], dtype=torch.float32)
    u_0 = model.p_rel.detach()
    u_0 = torch.tile(u_0, (n_batches, 1))

    # run opt routine
    # move to desired device
    inputs = inputs.to(device)
    targets = targets.to(device)
    h_0 = h_0.to(device)
    r_0 = r_0.to(device)
    u_0 = u_0.to(device)

    optimizer.zero_grad()

    # train model weights
    losses = list()
    params = list()
    state_vars = list()
    for trial_idx in range(n_trials):

        loss, init_params, state_vars_raw = train_bptt(
            inputs, targets, times, model, loss_fn,
            optimizer, h_0, r_0, u_0, dt=dt,
            include_stp=include_stp, noise_tau=noise_tau,
            noise_std=noise_std, include_corr_noise=include_corr_noise)

        if trial_idx in return_trials:
            params.append(init_params)

            ext_in, h_t, r_t, u_t, z_t = state_vars_raw
            # select 1st (and only) batch trial
            state_vars_processed = (ext_in.detach().numpy()[0],
                                    model.transfer_func(h_t).detach().numpy()[0],
                                    r_t.detach().numpy()[0],
                                    u_t.detach().numpy()[0],
                                    z_t.detach().numpy()[0])
            state_vars.append(state_vars_processed)

        losses.append(loss)
        print(f"Training trial {trial_idx} complete!")

    # investigate fitted model after training; test output using iid noise
    # averaged across 10 batch trials
    inputs_test = torch.tile(inputs, dims=(10, 1, 1))
    targets_test = torch.tile(targets, dims=(10, 1, 1))
    h_0_test = torch.tile(h_0, dims=(10, 1))
    r_0_test = torch.tile(r_0, dims=(10, 1))
    u_0_test = torch.tile(u_0, dims=(10, 1))
    state_vars_post_raw, sim_stats_post = test_and_get_stats(
        inputs_test, targets_test, times, model, loss_fn,
        h_0_test, r_0_test, u_0_test,
        include_stp=include_stp, noise_tau=noise_tau, noise_std=noise_std,
        include_corr_noise=include_corr_noise, plot=False)

    losses.append(sim_stats_post['loss'])

    if trial_idx + 1 in return_trials:
        final_params = [param.detach().numpy() for param in model.parameters()
                        if param.requires_grad]
        params.append(final_params)

        ext_in, h_t, r_t, u_t, z_t = state_vars_post_raw
        # select 1st batch trial
        state_vars_post_processed = (ext_in.detach().numpy()[0],
                                     model.transfer_func(h_t).detach().numpy()[0],
                                     r_t.detach().numpy()[0],
                                     u_t.detach().numpy()[0],
                                     z_t.detach().numpy()[0])
        state_vars.append(state_vars_post_processed)

    return losses, params, state_vars


if __name__ == '__main__':
    # set meta-parameters
    # for plotting style
    custom_params = {'axes.spines.right': False, 'axes.spines.top': False}
    sns.set_theme(style='ticks', rc=custom_params)
    # for pytorch tensors
    device = 'cpu'
    # for reproducibility while troubleshooting; numpy is for model sparse conns
    torch.random.manual_seed(93214)
    np.random.seed(35107)

    # set simulation parameters
    dt = 1e-3  # 1 ms
    tstop = 1.2  # 1 sec
    times = np.arange(-0.1 + dt, tstop + dt, dt)
    n_times = len(times)

    # create network and scale up gain to accomodate STP
    n_inputs, n_hidden, n_outputs = 1, 500, 10
    bad_net = False
    while bad_net is True:
        # instantiate network
        model = RNN(n_inputs=n_inputs, n_hidden=n_hidden,
                    n_outputs=n_outputs)

        # run gain adjustment; modifies gain in-place
        gains = adjust_gain_stp(model, times, n_steps=8)

        # check for an abberant drop in gain
        gain_diffs = np.diff(gains)
        increases = gain_diffs[gain_diffs > 0]
        decreases = gain_diffs[gain_diffs <= 0]
        if np.abs(decreases).max() > 2 * increases.max():
            bad_net = True

    # save baseline and adjusted gain values for later
    base_gain = gains[0]
    adjusted_gain = gains[-1]

    # instantiate loss function and optimizer; link to RNN output weights for
    # tuning
    mse_fn = nn.MSELoss()
    loss_fn = lambda a, b: mse_fn(a, b) / (b ** 2).mean()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # save inital state of tuned model parameters
    init_W_hz = model.W_hz.data.detach()
    init_offset_hz = model.offset_hz.data.detach()

    n_sims = len(sim_params_all)
    n_return_trials = len(return_trials)
    n_vars = 5
    losses_all = np.empty((n_sims, n_trials + 1), dtype=np.float32)
    model_params_all = {'W_hz': np.empty((n_sims, n_return_trials, n_outputs,
                                          n_hidden), dtype=np.float32),
                        'offset_hz': np.empty((n_sims, n_return_trials,
                                               n_outputs), dtype=np.float32)}
    state_vars_all = {'ext_in': np.empty((n_sims, n_return_trials, n_times,
                                          n_hidden), dtype=np.float32),
                      'h_t': np.empty((n_sims, n_return_trials, n_times,
                                       n_hidden), dtype=np.float32),
                      'r_t': np.empty((n_sims, n_return_trials, n_times,
                                       n_hidden), dtype=np.float32),
                      'u_t': np.empty((n_sims, n_return_trials, n_times,
                                       n_hidden), dtype=np.float32),
                      'z_t': np.empty((n_sims, n_return_trials, n_times,
                                       n_outputs), dtype=np.float32)}

    for sim_idx, sim_params in enumerate(sim_params_all):
        tau, include_stp, noise_tau, noise_std, include_corr_noise = sim_params

        # reset output weights
        with torch.no_grad():
            model.W_hz.copy_(init_W_hz)
            model.offset_hz.copy_(init_offset_hz)

        losses, model_params, state_vars = train_net(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            times=times,
            tau=tau,
            include_stp=include_stp,
            noise_tau=noise_tau,
            noise_std=noise_std,
            include_corr_noise=include_corr_noise,
            adjusted_gain=adjusted_gain,
            n_trials=n_trials,
            return_trials=return_trials,
            device=device)

        losses_all[sim_idx, :] = losses
        # convert to ragged array, then parse according to parameter name
        # model_params = np.array(model_params, dtype=object)
        model_params_all['W_hz'][sim_idx] = np.array([model_param[0] for model_param in model_params])
        model_params_all['offset_hz'][sim_idx] = np.array([model_param[1] for model_param in model_params])
        # convert to ragged array, then parse according to variable name
        state_vars_all['ext_in'][sim_idx] = np.array([state_var[0] for state_var in state_vars])
        state_vars_all['h_t'][sim_idx] = np.array([state_var[1] for state_var in state_vars])
        state_vars_all['r_t'][sim_idx] = np.array([state_var[2] for state_var in state_vars])
        state_vars_all['u_t'][sim_idx] = np.array([state_var[3] for state_var in state_vars])
        state_vars_all['z_t'][sim_idx] = np.array([state_var[4] for state_var in state_vars])

    # save simulation data in HDF5 file within output directory
    fname = op.join(output_dir, 'sim_data.hdf5')
    with h5py.File(fname, 'w') as f_write:
        f_write.create_dataset('sim_params', data=np.array(sim_params_all))
        for key, val in model_params_all.items():
            f_write.create_dataset(key, data=val)
        for key, val in state_vars_all.items():
            f_write.create_dataset(key, data=val)