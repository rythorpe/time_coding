"""Training and test functions for RNN model."""

import numpy as np
from scipy.spatial import distance
from sklearn import linear_model
import matplotlib.pyplot as plt

import torch

from utils import est_dimensionality
from viz import plot_state_traj


def step_sangers_rule(W, W_mask, inputs, outputs, lr=1e-5):
    '''Online unsupervised learning method implementing Sanger's Rule.'''
    xcov = torch.outer(outputs, inputs)
    autocov = torch.outer(outputs, outputs)
    # mask for non-zero connections
    dW = lr * (xcov - torch.tril(autocov) @ (W * W_mask)) * W_mask
    return dW


def pre_train(inputs, times, model, h_0, r_0, u_0):
    dt = times[1] - times[0]
    n_times = len(times)
    init_params = torch.cat((model.W_hh[model.W_hh_mask == 1],
                             model.W_hz.data.flatten()))
    model.train()

    # run model without storing gradients until t=0
    with torch.no_grad():
        h_t, r_t, u_t, z_t = model(inputs[:, times <= 0, :],
                                   h_0=h_0, r_0=r_0, u_0=u_0, dt=dt)

        # now, train using at each time point using Sanger's Rule
        step_size = 1
        t_0_idx = np.nonzero(times > 0)[0][0]
        for t_idx in np.arange(t_0_idx + step_size, n_times + step_size,
                               step_size):
            # compute prediction error
            t_minus_1_idx = t_idx - step_size
            h_0 = h_t[:, -1, :].detach()
            h_t, r_t, u_t, z_t = model(inputs[:, t_minus_1_idx:t_idx, :],
                                       h_0=h_0, r_0=r_0, u_0=u_0, dt=dt)
            model.W_hh.data += step_sangers_rule(model.W_hh.data,
                                                 model.W_hh_mask,
                                                 h_0[0], h_t[0, 0])

    updated_params = torch.cat((model.W_hh[model.W_hh_mask == 1],
                                model.W_hz.data.flatten()))
    param_dist = (torch.linalg.norm(updated_params - init_params)
                  / torch.linalg.norm(init_params))

    return param_dist.numpy(force=True)


def train_force(inputs, targets, times, model, loss_fn, optimizer,
                h_0, r_0, u_0, presyn_idx=0, debug_backprop=False):
    dt = times[1] - times[0]
    n_times = len(times)
    model.train()

    # run model without storing gradients until t=0
    with torch.no_grad():
        h_t, r_t, u_t, z_t = model(inputs[:, times <= 0, :],
                                   h_0=h_0, r_0=r_0, u_0=u_0, dt=dt)

    # if debug_backprop:
    #     dWhh_dloss_true = torch.empty(n_hidden, n_hidden,
    #                                   requires_grad=False)

    # now, train using FORCE
    step_size = 2
    losses = list()
    t_0_idx = np.nonzero(times > 0)[0][0]
    for t_idx in range(t_0_idx + step_size, n_times, step_size):
        # compute prediction error
        t_minus_1_idx = t_idx - step_size
        # set initial states to the last time point of the prior forward pass
        h_0 = h_t[:, -1, :].detach()
        if r_0 is not None:
            r_0 = r_t[:, -1, :].detach()
        if u_0 is not None:
            u_0 = u_t[:, -1, :].detach()
        h_t, r_t, u_t, z_t = model(inputs[:, t_minus_1_idx:t_idx, :],
                                   h_0=h_0, r_0=r_0, u_0=u_0, dt=dt)

        # loss at t - delta_t
        loss = loss_fn(z_t[:, -1, :], targets[:, t_idx, :])
        # backpropagation
        loss.backward()

        # if debug_backprop:
        #     dWhh_dloss_true = model.W_hh[:, :].copy().flatten()

        optimizer.step()
        optimizer.zero_grad()

        # reset presyn_scaling vector
        model.W_hh *= model.presyn_scaling.detach()
        torch.nn.init.ones_(model.presyn_scaling)

        losses.append(loss.item())

    return np.mean(losses)


def train_bptt_sparse(inputs, targets, times, model, loss_fn, optimizer,
                      h_0, r_0, u_0, p_backprop=0.2):
    dt = times[1] - times[0]
    n_times = len(times)
    model.train()
    optimizer.zero_grad()

    # run without calculating loss (and backprop) until t=0
    h_t, r_t, u_t, z_t = model(inputs[:, times <= 0, :],
                               h_0=h_0, r_0=r_0, u_0=u_0, dt=dt)

    losses = list()

    # backprop gradient from current time through each previous step
    t_0_idx = np.nonzero(times > 0)[0][0]
    n_time_after_0 = np.count_nonzero(times > 0)
    for t_idx in range(t_0_idx + 1, n_times, 1):
        # compute prediction error
        t_minus_1_idx = t_idx - 1
        # set initial states to the last time point of the prior forward pass
        h_0 = h_t[:, -1, :].detach()
        if r_0 is not None:
            r_0 = r_t[:, -1, :].detach()
        if u_0 is not None:
            u_0 = u_t[:, -1, :].detach()
        h_t, r_t, u_t, z_t = model(inputs[:, t_minus_1_idx:t_idx, :],
                                   h_0=h_0, r_0=r_0, u_0=u_0, dt=dt)
        # h_t_all.append(h_t)

        # loss at most recent time step
        # normalize by number of loss samples accumulated during backprop
        loss = (loss_fn(z_t[:, -1, :], targets[:, t_idx, :]) /
                (n_time_after_0 * p_backprop))

        # backprop only a proportion of the observed time points to promote
        # stability
        if np.random.rand() < p_backprop:
            loss.backward(retain_graph=True)
            losses.append(loss.item())

    optimizer.step()
    # reset presyn_scaling vector
    # model.W_hh *= model.presyn_scaling.detach()
    # torch.nn.init.ones_(model.presyn_scaling)
    optimizer.zero_grad()

    return np.sum(losses)


def train_bptt(inputs, targets, times, model, loss_fn, optimizer,
               h_0, r_0, u_0, dt, include_stp, noise_tau, noise_std, include_corr_noise):
    model.train()

    init_params = [param.detach().numpy() for param in model.parameters()
                   if param.requires_grad]
    state_vars = model(inputs, h_0=h_0, r_0=r_0, u_0=u_0, dt=dt,
                       include_stp=include_stp, noise_tau=noise_tau,
                       noise_std=noise_std, include_corr_noise=include_corr_noise)
    z_t = state_vars[4]
    loss = loss_fn(z_t[:, times > 0, :], targets[:, times > 0, :])
    loss.backward()

    optimizer.step()
    # scale W_hh sparsely and reset presyn_scaling vector
    # model.W_hh[:, :50] *= model.presyn_scaling.detach()[:50]
    # torch.nn.init.ones_(model.presyn_scaling)
    optimizer.zero_grad()

    return loss.item(), init_params, state_vars


def solve_ls_batch(hidden_sr, target_output):
    # assumes hidden_sr is trials x times x units
    n_trials, n_times, n_h_units = hidden_sr.shape
    _, _, n_out_units = target_output.shape
    # concat across trials (1st dim)
    hidden_sr = torch.reshape(hidden_sr, (n_trials * n_times,
                                          n_h_units))
    target_output = torch.reshape(target_output, (n_trials * n_times,
                                                  n_out_units))

    reg_model = linear_model.Ridge(alpha=1.0, fit_intercept=True)
    reg_model.fit(hidden_sr, target_output)
    weights = torch.tensor(reg_model.coef_, dtype=torch.float32)
    offsets = torch.tensor(reg_model.intercept_, dtype=torch.float32)

    return weights, offsets


def sim_batch(inputs, targets, times, model, loss_fn, h_0, r_0, u_0,
              dt, include_stp, noise_tau, noise_std,
              include_corr_noise):
    model.eval()

    with torch.no_grad():
        # simulate and calculate total output error
        ext_in, h_t, r_t, u_t, z_t = model(inputs, h_0=h_0, r_0=r_0, u_0=u_0,
                                           dt=dt, include_stp=include_stp,
                                           noise_tau=noise_tau,
                                           noise_std=noise_std,
                                           include_corr_noise=include_corr_noise)
        loss = loss_fn(z_t[:, times > 0, :], targets[:, times > 0, :])

    # select first batch trial
    state_vars = (ext_in.cpu(),
                  h_t.cpu(),
                  r_t.cpu(),
                  u_t.cpu(),
                  z_t.cpu())

    return state_vars, loss.item()


def test_and_get_stats(inputs, targets, times, model, loss_fn, h_0, r_0, u_0,
                       dt, include_stp, noise_tau, noise_std,
                       include_corr_noise, plot=True):
    model.eval()

    with torch.no_grad():
        # simulate and calculate total output error
        ext_in, h_t, r_t, u_t, z_t = model(inputs, h_0=h_0, r_0=r_0, u_0=u_0,
                                           dt=dt, include_stp=include_stp,
                                           noise_tau=noise_tau,
                                           noise_std=noise_std,
                                           include_corr_noise=include_corr_noise)
        loss = loss_fn(z_t[:, times > 0, :], targets[:, times > 0, :])

    try:
        print(f"Test loss: {loss.item():>7f}")
    except RuntimeError:
        Warning("Test loss isn't a scalar!")

    # select first batch trial to visualize single-trial trajectories
    noise_trial = ext_in.cpu()[0] - inputs.cpu()[0]
    hidden_trial = model.transfer_func(h_t).cpu()[0]
    syn_eff_trial = r_t.cpu()[0] * u_t.cpu()[0]
    outputs_trial = z_t.cpu()[0]
    targets_trial = targets.cpu()[0]

    # visualize network's response
    if plot:
        # for not, plot injected noise over time as perturbation
        fig = plot_state_traj(perturb=noise_trial, h_units=hidden_trial,
                              syn_eff=syn_eff_trial, outputs=outputs_trial,
                              targets=targets_trial, times=times)
        fig.show()

    # calculate metrics-of-interest
    n_dim = est_dimensionality(hidden_trial)
    stats = dict(loss=loss.item(), dimensionality=n_dim)

    # package all state variables
    state_vars = (ext_in.cpu(),
                  h_t.cpu(),
                  r_t.cpu(),
                  u_t.cpu(),
                  z_t.cpu())

    return state_vars, stats
