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
                   est_dimensionality, generate_noise)
from models import RNN
from train import sim_batch, train_bptt, test_and_get_stats
from viz import plot_state_traj, plot_all_units

###
p_rel_std_vals = [0.15, 0.05]
###
params_between_net = list()
params_between_net_keys = ['p_rel_std']
for p_rel_std in p_rel_std_vals:
    # p_rel_std
    params_between_net.append([p_rel_std])

###
noise_tau_vals = [0.1]
noise_std_vals = [0.1]
beta_vals = [0., 80.]
n_targ_seq_vals = [1, 2, 3]
seq_compression_vals = [0.5, 0.75, 1.0]
###
params_train = list()
params_train_keys = ['beta', 'noise_tau', 'noise_std', 'n_targ_seq', 'seq_compression']
for beta in beta_vals:
    for noise_tau in noise_tau_vals:
        for noise_std in noise_std_vals:
            for n_targ_seq in n_targ_seq_vals:
                for seq_compression in seq_compression_vals:
                    # beta, noise_tau, noise_std, n_targ_seq, seq_compression
                    params_train.append([beta, noise_tau, noise_std, n_targ_seq, seq_compression])

params_test = list()
params_test_keys = ['noise_tau_test', 'noise_std_test']
# NB: we current test on each value of noise we train on
for noise_tau in noise_tau_vals:
    for noise_std in noise_std_vals:
        # noise_tau, noise_std
        params_test.append([noise_tau, noise_std])

n_random_nets = 10
n_jobs = 10
n_test_trials = 10
output_dir = '/projects/ryth7446/time_coding_output'
# n_random_nets = 2
# n_jobs = 4
# n_test_trials = 10
# output_dir = '/home/ryan/Desktop'


def test_trained_net(evoked_input, targets, times, model, loss_fn,
                     h_0, r_0, u_0, dt, noise_tau, noise_std,
                     plot=False, n_test_trials=10, inputs_to_plot=None):
    '''Train current instantiation of network model.

    Returns network parameters and variables for a batch of sim trials.
    '''

    n_batch_trials, n_times, _ = evoked_input.shape
    n_hidden, n_outputs = model.n_hidden, model.n_outputs
    noise = generate_noise(n_batch_trials * n_test_trials, times, n_hidden,
                           noise_tau, noise_std, dt=1e-4)
    # tile across test trials
    inputs = torch.tile(evoked_input, dims=(n_test_trials, 1, 1)) + noise
    targets = torch.tile(targets, dims=(n_test_trials, 1, 1))
    h_0 = torch.tile(h_0, dims=(n_test_trials, 1))
    r_0 = torch.tile(r_0, dims=(n_test_trials, 1))
    u_0 = torch.tile(u_0, dims=(n_test_trials, 1))

    # ensure tensors have been moved to desired device prior to forward-pass
    device = 'cpu'
    model.to(device)
    inputs = inputs.to(device)
    targets = targets.to(device)
    h_0 = h_0.to(device)
    r_0 = r_0.to(device)
    u_0 = u_0.to(device)

    model.eval()
    with torch.no_grad():
        # simulate network
        h_t, r_t, u_t, z_t = model(inputs, h_0=h_0, r_0=r_0, u_0=u_0, dt=dt)

    hidden_sr = model.transfer_func(h_t).detach()
    syn_eff = r_t.detach() * u_t.detach()
    output = z_t.detach()

    mse = list()
    training_batch_dims = list()
    mean_rates = list()
    mean_syn_effs = list()
    for train_dim_idx in range(n_batch_trials):

        hidden_sr_train_dim = hidden_sr[train_dim_idx::n_batch_trials, ...]
        syn_eff_train_dim = syn_eff[train_dim_idx::n_batch_trials, ...]
        output_train_dim = output[train_dim_idx::n_batch_trials, ...]
        targets_train_dim = targets[train_dim_idx::n_batch_trials, ...]

        peak_idxs = targets_train_dim[0, :, :].argmax(dim=0)
        seq_t_mask = np.zeros((n_times,))
        seq_t_mask[peak_idxs[0]:peak_idxs[-1]] = 1

        # mean spike rate across trials and time (keep unit dim)
        mean_rates.append(hidden_sr_train_dim[:, seq_t_mask == 1, :].mean(dim=(0, 1)))
        # mean synaptic efficacy across trials and time (keep unit dim)
        mean_syn_effs.append(syn_eff_train_dim[:, seq_t_mask == 1, :].mean(dim=(0, 1)))

        # final MSE
        mse.append(loss_fn(output_train_dim[:, times > 0, :],
                           targets_train_dim[:, times > 0, :]).item())

        # dimensionality of hidden unit responses across test trials for a given training batch trial
        test_trial_stack = []
        for trial_idx in range(n_test_trials):
            test_trial_stack.append(hidden_sr_train_dim[trial_idx, ...])
        test_trial_stack = torch.cat(test_trial_stack, dim=1)
        training_batch_dims.append(est_dimensionality(test_trial_stack))
    avg_train_batch_dim = np.mean(training_batch_dims)

    # dimensionality of hidden unit responses across all test and training batch trials
    all_trial_stack = []
    for trial_idx in range(n_batch_trials * n_test_trials):
        all_trial_stack.append(hidden_sr[trial_idx, ...])
    all_trial_stack = torch.cat(hidden_sr, dim=1)
    batch_dim = est_dimensionality(all_trial_stack)

    metrics = {'mean_rate': mean_rates, 'mean_syn_eff': mean_syn_effs,
               'mse': mse, 'avg_dim_index_test_trial_agg': avg_train_batch_dim,
               'dim_index_all_trial_agg': batch_dim}

    if plot is True:
        # plot last batch trial, which should be the one with the shortest
        # temporal sequence
        if inputs_to_plot is None:
            ext_in_trial = inputs.detach().numpu()[-1]
        else:
            ext_in_trial = inputs_to_plot.detach().numpy()[-1]
        hidden_sr_trial = model.transfer_func(h_t).detach().numpy()[-1]
        syn_eff_trial = r_t.detach().numpy()[-1] * u_t.detach().numpy()[-1]
        outputs_trial = z_t.detach().numpy()[-1]
        targets_trial = targets.detach().numpy()[-1]

        fig_traj = plot_state_traj(perturb=ext_in_trial,
                                   h_units=hidden_sr_trial,
                                   syn_eff=syn_eff_trial,
                                   outputs=outputs_trial,
                                   targets=targets_trial,
                                   times=times)

        # sort hidden units according to peak activity for plotting
        peak_t_idx = list()
        for h_idx in range(n_hidden):
            # select 1st time index the threshhold is cross
            max_idx = hidden_sr_trial[times > 0, h_idx].argmax()
            peak_t_idx.append(max_idx)
        sort_idxs = np.argsort(peak_t_idx)

        fig_state = plot_all_units(h_units=hidden_sr_trial[:, sort_idxs],
                                   syn_eff=syn_eff_trial[:, sort_idxs],
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

    p_rel_std = param_net[0]

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

    print(f'begin training session for net instance {net_idx}')

    # instantiate random network; resample if training is unstable
    sample_new_net = True
    while sample_new_net is True:
        # instantiate network
        model = RNN(n_hidden=n_hidden, n_outputs=n_outputs,
                    p_rel_std=p_rel_std)

        # define input to network
        # these will be held constant across training conditions
        evoked_input_timeseries = torch.zeros((3, n_times, n_hidden))
        perturb_win_mask = times >= 0
        single_unit_input = generate_noise(3, times[perturb_win_mask], 1,
                                           noise_tau=0.5, noise_std=0.5)
        evoked_input_timeseries[:, perturb_win_mask, :] = torch.randn(n_hidden) * single_unit_input

        # save initial network parameters
        learned_params_init = {
            'offset_ih': model.offset_ih.data.detach().clone(),
            'W_hh': model.W_hh.data.detach().clone(),
            'W_hh_mask': model.W_hh_mask.detach().clone(),  # static
            'W_hz': model.W_hz.data.detach().clone(),
            'W_hz_mask': model.W_hz_mask.detach().clone(),  # static
            'p_rel': model.p_rel.detach().clone()  # static
            }
        for key, val in learned_params_init.items():
            file.create_dataset(key, data=val)

        # instantiate optimizer (with refs to model params undergoing training)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # train network
        for training_cond_idx, param_train in enumerate(params_train):

            # set controlled training (pre-learning) params
            # NB: beta controls strength of STP
            beta, noise_tau, noise_std, n_targ_seq, seq_compression = param_train

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

            model.beta = beta

            # selected appropriate number of evoked inputs
            n_batch_trials = n_targ_seq
            evoked_input = evoked_input_timeseries[:n_batch_trials]

            # this will be used for plotting later
            common_evoked_input = torch.zeros((n_batch_trials, n_times, n_hidden))
            common_evoked_input[:, perturb_win_mask, :] = torch.tile(single_unit_input[:n_batch_trials, :, :], (1, 1, n_hidden))

            # generate big batch of OU process timeseries at the beginning to
            # draw from during training
            n_rand_trials = n_batch_trials * 3
            n_rand_units = n_hidden * 3
            noise_batch = generate_noise(n_rand_trials, times, n_rand_units,
                                         noise_tau, noise_std, dt=1e-4)  # n_trials, n_times, n_dim

            # define output targets, one set of Gaussian peaks for each batch trial
            # set std s.t. amplitude decays to 1/e at intersection with next target
            # tile center of target delays spanning sim duration (minus margins)
            targets = torch.zeros(n_targ_seq, n_times, n_outputs)
            # largest to smallest; default to largest delay when n_targ_seq=1
            last_delays = np.linspace(1.0, 1.0 * seq_compression, n_targ_seq)
            for seq_idx in range(n_targ_seq):
                last_delay = last_delays[seq_idx]
                delay_times = np.linspace(last_delay / n_outputs, last_delay, n_outputs)
                targets[seq_idx, ...] = get_gaussian_targets(
                    1, delay_times, times,
                    last_delay / (n_outputs * 2 * np.sqrt(2))
                )

            # set initial conditions of recurrent units fixed across iterations of
            # training and testing
            h_0 = torch.zeros(n_hidden)
            h_0 = torch.tile(h_0, (n_batch_trials, 1))  # replicate for each batch
            r_0 = torch.ones(n_hidden)
            r_0 = torch.tile(r_0, (n_batch_trials, 1))
            u_0 = model.p_rel.detach()
            u_0 = torch.tile(u_0, (n_batch_trials, 1))

            # ensure tensors are located on appropriate device
            device = 'cpu'
            model.to(device)
            targets = targets.to(device)
            h_0 = h_0.to(device)
            r_0 = r_0.to(device)
            u_0 = u_0.to(device)

            # train network weights
            n_training_trials = 2500
            loss_per_iter = list()
            for _ in range(n_training_trials):
                rand_trial_idxs = torch.randperm(n_rand_trials)[:n_batch_trials]
                rand_units_idxs = torch.randperm(n_rand_units)[:n_hidden]
                noise = noise_batch[rand_trial_idxs, ...]
                noise = noise[:, :, rand_units_idxs]
                inputs = evoked_input + noise
                inputs = inputs.to(device)
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
                'W_hz': model.W_hz.data.detach().clone()
                }
            for key, val in learned_params_final.items():
                training_grp.create_dataset(key, data=val)

            # now, test trained network and save metrics
            metrics_appended = defaultdict(list)
            for param_test in params_test:

                noise_tau_test, noise_std_test = param_test

                # select subset of nets/conditions to plot and save example sims
                plot_instance = net_idx < len(params_between_net) * 2

                metrics, figs = test_trained_net(
                    evoked_input=evoked_input,
                    targets=targets,
                    times=times,
                    model=model,
                    loss_fn=loss_fn,
                    h_0=h_0,
                    r_0=r_0,
                    u_0=u_0,
                    dt=dt,
                    noise_tau=noise_tau_test,
                    noise_std=noise_std_test,
                    plot=plot_instance,
                    n_test_trials=n_test_trials,
                    inputs_to_plot=common_evoked_input,
                    )
                for key, val in metrics.items():
                    metrics_appended[key].append(val)

                if plot_instance is True:
                    fname_traj_fig = f'fig_ts_net{net_idx:02d}_beta{beta:02.1f}_n_targs{n_targ_seq:1d}_seq_compr{seq_compression:.2f}.pdf'
                    figs[0].savefig(op.join(output_dir, fname_traj_fig))
                    plt.close(figs[0])
                    fname_state_fig = f'fig_state_net{net_idx:02d}_beta{beta:02.1f}_n_targs{n_targ_seq:1d}_seq_compr{seq_compression:.2f}.pdf'
                    figs[1].savefig(op.join(output_dir, fname_state_fig))
                    plt.close(figs[1])

            for test_param_idx, test_param_key in enumerate(params_test_keys):
                test_param_vals = np.array(params_test)[:, test_param_idx]
                training_grp.create_dataset(test_param_key,
                                            data=test_param_vals)

            for key, val in metrics_appended.items():
                training_grp.create_dataset(key, data=val)

    print(f'training + eval of net instance {net_idx} complete')


if __name__ == '__main__':

    # seed state for reproducibility; numpy is for model sparse conns
    torch.random.manual_seed(93214)
    np.random.seed(35107)

    params_between_net_rep = np.tile(params_between_net, (n_random_nets, 1))
    n_total_nets = params_between_net_rep.shape[0]

    Parallel(n_jobs=n_jobs)(delayed(eval_net_instance)
                            (params_between_net_rep[net_idx], params_train,
                             params_test, net_idx)
                            for net_idx in range(n_total_nets))
