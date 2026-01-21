import os.path as op
from collections import defaultdict

import h5py
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from scipy import optimize

import torch
from torch import nn

from utils import (get_gaussian_targets, get_commit_hash, get_timestamp,
                   est_dimensionality)
from models import RNN
from train import sim_batch, train_bptt, test_and_get_stats
from viz import plot_state_traj, plot_all_units


noise_tau_vals = 10 ** np.linspace(-2, 0, 3)
noise_std_vals = np.linspace(1e-1, 3e-1, 3)
beta_vals = np.linspace(0, 50, 3)
p_rel_range_vals = [2]

params_between_net = list()
params_between_net_keys = ['p_rel_range']
for p_rel_range in p_rel_range_vals:
    # p_rel_range
    params_between_net.append([p_rel_range])

params_train = list()
params_train_keys = ['beta', 'noise_tau', 'noise_std']
for beta in beta_vals:
    for noise_tau in noise_tau_vals:
        for noise_std in noise_std_vals:
            # beta, noise_tau, noise_std
            params_train.append([beta, noise_tau, noise_std])

params_test = list()
params_test_keys = ['noise_tau_test', 'noise_std_test']
for noise_tau in noise_tau_vals:
    for noise_std in noise_std_vals:
        # noise_tau, noise_std
        params_test.append([noise_tau, noise_std])

n_random_nets = 20
n_jobs = 32
n_test_trials = 20
output_dir = '/projects/ryth7446/time_coding_output'
# n_random_nets = 2
# n_jobs = 2
# n_test_trials = 20
# output_dir = '/home/ryan/time_coding/data'
# output_dir = '/home/ryan/Desktop'


def adjust_gain_stp(model, times, dt, n_steps=8):
    '''Adjust network gain to compensate for STP in-place.

    Returns initial gain value.
    '''

    # define inputs (for contextual modulation / recurrent perturbations)
    n_trials = 1  # simulates a single trial between adjustments
    n_hidden = model.n_hidden
    n_outputs = model.n_outputs
    n_times = len(times)
    inputs = torch.zeros((1, n_times, 1))
    perturb_dur = 0.05  # 50 ms
    perturb_win_mask = np.logical_and(times > -perturb_dur, times < 0)
    inputs[:, perturb_win_mask, :] = 1.0

    # define output targets
    # set std s.t. amplitude decays to 1/e at intersection with next target
    targ_std = 0.05 / np.sqrt(2)  # ~35 ms
    # tile center of target delays spanning sim duration (minus margins)
    delay_times = np.linspace(1 / n_outputs, 1.0, n_outputs)
    targets = get_gaussian_targets(1, delay_times, times, targ_std)

    # set initial conditions of recurrent units fixed across iterations of
    # training and testing
    h_0 = torch.zeros(n_hidden)  # steady-state for postsyn activity var
    h_0 = torch.tile(h_0, (n_trials, 1))  # replicate for each batch
    r_0 = torch.ones(n_hidden)  # steady-state for depression var
    r_0 = torch.tile(r_0, (n_trials, 1))
    u_0 = model.p_rel.detach()  # steady-state for facilitation var
    u_0 = torch.tile(u_0, (n_trials, 1))

    # run opt routine
    # move to desired device
    device = 'cpu'
    inputs = inputs.to(device)
    targets = targets.to(device)
    h_0 = h_0.to(device)
    r_0 = r_0.to(device)
    u_0 = u_0.to(device)

    # move to desired device
    model.to(device)

    init_beta = model.beta

    # (re)set gain to its baseline value
    model.gain = model._init_gain

    # slowly inclrease STP beta param, turning up gain adjustment accordingly
    gain_vals = list()
    for beta_fraction in torch.linspace(0, 1, n_steps + 1):
        model.beta = init_beta * beta_fraction

        # adjust beta twice for each incremental increase in beta
        for update_idx in range(2):
            state_vars_, _ = sim_batch(inputs, targets, times, model, loss_fn,
                                       h_0, r_0, u_0, dt=dt,
                                       noise_tau=0.01,
                                       noise_std=0.0)
            hidden_sr_t_, r_t_, u_t_, output_sr_t_ = state_vars_
            syn_eff_ = r_t_ * u_t_
            adjustment_fctr = model.p_rel.mean() / syn_eff_.mean()
            model.gain = model._init_gain * adjustment_fctr
            gain_vals.append(model.gain.item())

    return gain_vals


def test_trained_net(inputs, targets, times, model, loss_fn,
                     h_0, r_0, u_0, dt, noise_tau, noise_std,
                     include_corr_noise=False, plot=False):
    '''Train current instantiation of network model.

    Returns network parameters and variables for a batch of sim trials.
    '''

    # ensure tensors have been moved to desired device prior to forward-pass
    device = 'cpu'
    model.to(device)
    inputs = inputs.to(device)
    targets = targets.to(device)
    h_0 = h_0.to(device)
    r_0 = r_0.to(device)
    u_0 = u_0.to(device)

    n_trials, n_times, _ = inputs.shape
    n_inputs, n_hidden, n_outputs = (model.n_inputs, model.n_hidden,
                                     model.n_outputs)

    mse_vals = list()
    for trial_idx in range(n_trials):
        noise_ensembles = torch.ones(n_outputs)
        # assign left-out ensemble (w/o noise) round-robin
        noise_ensembles[trial_idx % n_outputs] = 0

        state_vars_raw = sim_batch(
            inputs=inputs[trial_idx:trial_idx + 1, ...],
            model=model,
            h_0=h_0[trial_idx:trial_idx + 1, ...],
            r_0=r_0[trial_idx:trial_idx + 1, ...],
            u_0=u_0[trial_idx:trial_idx + 1, ...],
            dt=dt,
            noise_tau=noise_tau,
            noise_std=noise_std,
            include_corr_noise=include_corr_noise,
            noise_ensembles=noise_ensembles
            )

        # calculate avg features of simulated data across batch trials
        h_t, r_t, u_t, z_t = state_vars_raw
        hidden_sr_test = model.transfer_func(h_t).detach()
        syn_eff_test = r_t.detach() * u_t.detach()
        # error
        read_out_ensemble_mask = noise_ensembles == 0
        z_t_read_out = z_t[:, :, read_out_ensemble_mask]
        targets_read_out = targets[trial_idx:trial_idx + 1, :, read_out_ensemble_mask]
        mse = loss_fn(z_t_read_out[:, times > 0, :],
                      targets_read_out[:, times > 0, :])
        mse_vals.append(mse)
        # # dimensionality of hidden unit responses
        # trial_dims = list()
        # for batch_trial in hidden_sr_test:
        #     trial_dims.append(est_dimensionality(batch_trial[times > 0, :]))
        # batch_dim = np.mean(trial_dims)
        # # mean spike rate across time
        # mean_rate = hidden_sr_test[:, times > 0, :].mean()
        # # mean synaptic efficacy across time
        # mean_syn_eff = syn_eff_test[:, times > 0, :].mean()

        # metrics = {'mse': mse, 'dim_index': batch_dim, 'mean_rate': mean_rate,
        #         'mean_syn_eff': mean_syn_eff}
    metrics = {'mse': np.mean(mse_vals)}

    if plot is True:
        ext_in_trial = (inputs[-1] @ model.W_ih.T + model.offset_ih + n_t[0]).detach().numpy()
        hidden_sr_trial = model.transfer_func(h_t).detach().numpy()[0]
        syn_eff_trial = r_t.detach().numpy()[0] * u_t.detach().numpy()[0]
        outputs_trial = z_t.detach().numpy()[0]
        targets_trial = targets.detach().numpy()[0]

        fig_traj = plot_state_traj(perturb=ext_in_trial,
                                   h_units=hidden_sr_trial,
                                   syn_eff=syn_eff_trial,
                                   outputs=outputs_trial,
                                   targets=targets_trial,
                                   times=times)
        fig_state = plot_all_units(h_units=hidden_sr_trial,
                                   syn_eff=syn_eff_trial,
                                   outputs=outputs_trial,
                                   targets=targets_trial,
                                   times=times)
        figs = (fig_traj, fig_state)
    else:
        figs = None

    return metrics, figs


def loss_fn(output, target):
    mse_fn = nn.MSELoss()
    return mse_fn(output, target) / (target ** 2).mean()


def eval_net_instance(param_net, params_train, params_test, net_idx):
    '''Sweeps over training conditions for a given random net, then runs tests
     each trained net and saves important metrics for each test condition.'''
    
    p_rel_range = param_net[0]

    
    # create HDF5 file for saving results
    fname_local = ('data_' + f'net{net_idx:02d}_' +
                   get_commit_hash() + '_' + get_timestamp() + '.hdf5')
    fname_absolute = op.join(output_dir, fname_local)
    file = h5py.File(fname_absolute, 'a')
    for net_param_idx, net_param_key in enumerate(params_between_net_keys):
        file.create_dataset(net_param_key, data=param_net[net_param_idx])

    # set simulation parameters
    dt = 5e-3  # 1 ms
    tstop = 1.2  # 1 sec
    times = np.arange(-0.1 + dt, tstop + dt, dt)
    n_times = len(times)

    # define network hyperparameters
    n_hidden, n_outputs = 500, 10
    if p_rel_range == 2:
        p_rel_range = (0.1, 0.6)  # high heterogeneity
    elif p_rel_range == 1:
        p_rel_range = (0.3, 0.4)  # low heterogeneity
    elif p_rel_range == 0:
        p_rel_range = (0.35, 0.35)  # homogeneous

    # define input to network
    n_trials = 1
    inputs = torch.zeros((n_trials, n_times, 1))
    perturb_dur = 0.05  # 50 ms
    perturb_win_mask = np.logical_and(times >= -perturb_dur, times < 0)
    inputs[:, perturb_win_mask, :] = 1.0

    # define output targets
    # set std s.t. amplitude decays to 1/e at intersection with next target
    targ_std = 0.05 / np.sqrt(2)  # ~35 ms
    # tile center of target delays spanning sim duration (minus margins)
    delay_times = np.linspace(1 / n_outputs, 1.0, n_outputs)
    targets = get_gaussian_targets(n_trials, delay_times, times, targ_std)

    print(f'begin training session for net instance {net_idx}')

    # instantiate random network; resample if training is unstable
    sample_new_net = True
    while sample_new_net is True:
        # instantiate network
        model = RNN(n_hidden=n_hidden, n_outputs=n_outputs,
                    p_rel_range=p_rel_range)

        # save initial network parameters
        learned_params_init = {
            'offset_ih': model.offset_ih.data.detach().clone(),
            'W_hh': model.W_hh.data.detach().clone(),
            'W_hh_mask': model.W_hh_mask.detach().clone(),  # static
            'W_hz': model.W_hz.data.detach().clone(),
            'W_hz_mask': model.W_hz_mask.detach().clone(),  # static
            'offset_hz': model.offset_hz.data.detach().clone(),
            'p_rel': model.p_rel.detach().clone()  # static
            }
        for key, val in learned_params_init.items():
            file.create_dataset(key, data=val)

        # instantiate optimizer (with refs to model params undergoing training)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

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

        # ensure tensors are located on appropriate device
        device = 'cpu'
        model.to(device)
        inputs = inputs.to(device)
        targets = targets.to(device)
        h_0 = h_0.to(device)
        r_0 = r_0.to(device)
        u_0 = u_0.to(device)

        # train network
        for training_cond_idx, param_train in enumerate(params_train):

            # set controlled training (pre-learning) params
            # NB: beta controls strength of STP
            beta, noise_tau, noise_std = param_train

            # create subdirectory (i.e., HDF5 'group') in which to save results
            # for this realization of the trained network
            training_grp = file.create_group(f'training_cond{training_cond_idx}')
            for training_param_idx, training_param_key in enumerate(params_train_keys):
                training_grp.create_dataset(training_param_key, data=param_train[training_param_idx])

            # (re)set network weights
            with torch.no_grad():
                model.offset_ih.copy_(learned_params_init['offset_ih'])
                model.W_hh.copy_(learned_params_init['W_hh'])
                model.W_hz.copy_(learned_params_init['W_hz'])
                model.offset_hz.copy_(learned_params_init['offset_hz'])

            model.beta = beta

            # train network weights
            n_training_trials = 3000
            # noise_tau = dt  # train with Gaussian white noise by setting noise_tau -> dt
            # noise_std = 1e-2
            loss_per_iter = list()
            for trial_idx in range(n_training_trials):

                loss, _, _ = train_bptt(
                    inputs, targets, times, model, loss_fn, optimizer,
                    h_0, r_0, u_0, dt=dt
                    )
                loss_per_iter.append(loss)

            # get loss after final update
            # plot model output after training
            _, sim_stats_post = test_and_get_stats(
                inputs, targets, times, model, loss_fn, h_0, r_0, u_0, dt=dt,
                plot=False
                )
            final_loss = sim_stats_post['loss']
            if np.isfinite(final_loss):
                sample_new_net = False
            else:
                print('warning: resampling network')
                break
            loss_per_iter.append(final_loss)
            # save loss trajectory
            training_grp.create_dataset('loss', data=loss_per_iter)
            
            # save final trained network parameters
            learned_params_final = {
                'offset_ih': model.offset_ih.data.detach().clone(),
                'W_hh': model.W_hh.data.detach().clone(),
                'W_hz': model.W_hz.data.detach().clone(),
                'offset_hz': model.offset_hz.data.detach().clone()
                }
            for key, val in learned_params_final.items():
                training_grp.create_dataset(key, data=val)

            # now, test trained network and save metrics
            metrics_appended = defaultdict(list)
            # pow_spec = list()
            for param_test in params_test:
                noise_tau_test, noise_std_test = param_test

                inputs_batch = torch.tile(inputs, dims=(n_test_trials, 1, 1))
                targets_batch = torch.tile(targets, dims=(n_test_trials, 1, 1))
                h_0_batch = torch.tile(h_0, dims=(n_test_trials, 1))
                r_0_batch = torch.tile(r_0, dims=(n_test_trials, 1))
                u_0_batch = torch.tile(u_0, dims=(n_test_trials, 1))
                
                # select subset of conditions to plot and save example sims
                plot = (noise_tau in [1e-2, 1e0] and
                        noise_tau_test == noise_tau and
                        noise_std in [1e-1, 2e-1] and
                        noise_std_test == noise_std_test and
                        model.beta in [0.0, 50.0] and
                        net_idx < 1)

                metrics, figs = test_trained_net(
                    inputs=inputs_batch,
                    targets=targets_batch,
                    times=times,
                    model=model,
                    loss_fn=loss_fn,
                    h_0=h_0_batch,
                    r_0=r_0_batch,
                    u_0=u_0_batch,
                    dt=dt,
                    plot=plot
                    )
                for key, val in metrics.items():
                    metrics_appended[key].append(val)
                # pow_spec.append(metrics['pow_spec'])
                if plot is True:
                    fname_traj_fig = f'fig_ts_net{net_idx:02d}_beta{beta:.2f}_std{noise_std:.2f}_tau{noise_tau:.2f}.png'
                    figs[0].savefig(op.join(output_dir, fname_traj_fig))
                    plt.close(figs[0])
                    fname_state_fig = f'fig_state_net{net_idx:02d}_beta{beta:.2f}_std{noise_std:.2f}_tau{noise_tau:.2f}.png'
                    figs[1].savefig(op.join(output_dir, fname_state_fig))
                    plt.close(figs[1])


            for test_param_idx, test_param_key in enumerate(params_test_keys):
                test_param_vals = np.array(params_test)[:, test_param_idx]
                training_grp.create_dataset(test_param_key,
                                            data=test_param_vals)

            for key, val in metrics_appended.items():
                training_grp.create_dataset(key, data=val)
            # training_grp.create_dataset('pow_spec', data=dim_index)

    print(f'training + eval of net instance {net_idx} complete')


if __name__ == '__main__':

    # seed state for reproducibility; numpy is for model sparse conns
    torch.random.manual_seed(93214)
    np.random.seed(35107)

    params_between_net = np.tile(params_between_net, (n_random_nets, 1))
    n_total_nets = params_between_net.shape[0]

    Parallel(n_jobs=n_jobs)(delayed(eval_net_instance)
                            (params_between_net[net_idx], params_train,
                             params_test, net_idx)
                            for net_idx in range(n_total_nets))
