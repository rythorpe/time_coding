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


def pre_train(inputs, times, model, h_0, r_0, u_0, dt,
              noise_tau, noise_std, include_corr_noise):
    n_times = len(times)
    model.eval()

    # run model without storing gradients until t=0
    with torch.no_grad():
        ext_in, h_t, r_t, u_t, z_t = model(inputs[:, times <= 0, :],
                                           h_0=h_0, r_0=r_0, u_0=u_0, dt=dt,
                                           noise_tau=noise_tau,
                                           noise_std=noise_std,
                                           include_corr_noise=include_corr_noise)

        # now, train using at each time point using Sanger's Rule
        step_size = 1
        t_0_idx = np.nonzero(times > 0)[0][0]
        for t_idx in np.arange(t_0_idx + step_size, n_times + step_size,
                               step_size):
            # compute prediction error
            t_minus_1_idx = t_idx - step_size
            h_0 = h_t[:, -1, :].detach()
            ext_in, h_t, r_t, u_t, z_t = model(
                inputs[:, t_minus_1_idx:t_idx, :],
                h_0=h_0, r_0=r_0, u_0=u_0, dt=dt,
                noise_tau=noise_tau, noise_std=noise_std,
                include_corr_noise=include_corr_noise)
            previous_rates = model.transfer_func(h_0[0]).detach().clone()
            current_rates = model.transfer_func(h_t[0, 0, :]).detach().clone()
            model.W_hh.data += step_sangers_rule(model.W_hh.data,
                                                 model.W_hh_mask,
                                                 previous_rates, current_rates)


class RLS:
    def __init__(self, n_vars, lambda_factor=1.0, P0_diag=1):
        self.n_vars = n_vars
        self.lambda_factor = lambda_factor
        self.P = P0_diag * torch.eye(n_vars) # Initial inverse correlation matrix

    def step(self, x, theta, theta_sel):
        # x: current regressor vector (feature vector)
        # y: current observed output
        
        x = x.reshape(-1, 1) # Ensure x is a column vector

        # Calculate gain vector
        K = (self.P @ x) / (self.lambda_factor + x.T @ self.P @ x)

        # Update subset parameter estimate
        # error = theta.grad[theta_sel[:, 0], theta_sel[:, 1]]
        with torch.no_grad():
            weighted_err = (K * theta.grad)[theta_sel[:, 0], theta_sel[:, 1]]
            # theta_ = theta + K * error
            # theta.copy_(theta_.detach())
            theta[theta_sel[:, 0], theta_sel[:, 1]] += weighted_err

        # Update inverse correlation matrix
        self.P = (1 / self.lambda_factor) * (self.P - K @ x.T @ self.P)


def train_bptt(inputs, targets, times, model, loss_fn, optimizer,
               h_0, r_0, u_0, dt, noise_tau, noise_std,
               include_corr_noise=False):
    model.train()

    init_params = [param.detach().numpy() for param in model.parameters()
                   if param.requires_grad]
    state_vars = model(inputs, h_0=h_0, r_0=r_0, u_0=u_0, dt=dt,
                       noise_tau=noise_tau, noise_std=noise_std,
                       include_corr_noise=include_corr_noise)
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


def sim_batch(inputs, model, h_0, r_0, u_0,
              dt, noise_tau, noise_std,
              include_corr_noise=False, noise_ensembles='all'):
    model.eval()

    n_trials, n_times, _ = inputs.shape

    if noise_ensembles == 'all':
        noise_ensembles = torch.ones(model.n_outputs)

    n_0_mask = noise_ensembles @ model.W_hz_mask
    n_0 = torch.zeros_like(h_0)

    n_t_all = torch.zeros(n_trials, n_times, model.n_hidden)
    r_t_all = torch.zeros(n_trials, n_times, model.n_hidden)
    u_t_all = torch.zeros(n_trials, n_times, model.n_hidden)
    h_t_all = torch.zeros(n_trials, n_times, model.n_hidden)
    z_t_all = torch.zeros(n_trials, n_times, model.n_outputs)

    with torch.no_grad():
        # simulate and calculate total output error
        for t_idx in range(n_times):
            I = inputs[:, t_idx:t_idx + 1, :]
            n_t, h_t, r_t, u_t, z_t = model(I, h_0=h_0, r_0=r_0, u_0=u_0,
                                            n_0=n_0,
                                            dt=dt, noise_tau=noise_tau,
                                            noise_std=noise_std,
                                            include_corr_noise=include_corr_noise)
            
            # zero-out noise in select ensembles
            n_t_masked = n_0_mask * n_t[:, -1, :]

            # save current state for output
            n_t_all[:, t_idx, :] = n_t_masked
            h_t_all[:, t_idx, :] = h_t[:, -1, :]
            r_t_all[:, t_idx, :] = r_t[:, -1, :]
            u_t_all[:, t_idx, :] = u_t[:, -1, :]
            z_t_all[:, t_idx, :] = z_t[:, -1, :]

            # set initial state for next time step
            n_0 = n_t_masked
            h_0 = h_t[:, -1, :]
            r_0 = r_t[:, -1, :]
            u_0 = u_t[:, -1, :]

    return n_t_all, h_t_all, r_t_all, u_t_all, z_t_all


def test_and_get_stats(inputs, targets, times, model, loss_fn, h_0, r_0, u_0,
                       dt, noise_tau, noise_std,
                       include_corr_noise, plot=True):
    model.eval()

    with torch.no_grad():
        # simulate and calculate total output error
        n_t, h_t, r_t, u_t, z_t = model(inputs, h_0=h_0, r_0=r_0, u_0=u_0,
                                           dt=dt, noise_tau=noise_tau,
                                           noise_std=noise_std,
                                           include_corr_noise=include_corr_noise)
        h_sr = model.transfer_func(h_t)
        # loss = loss_fn(h_sr[:, times > 0, :], targets[:, times > 0, :])
        loss = loss_fn(z_t[:, times > 0, :], targets[:, times > 0, :])

    try:
        print(f"Test loss: {loss.item():>7f}")
    except RuntimeError:
        Warning("Test loss isn't a scalar!")

    # select first batch trial to visualize single-trial trajectories
    ext_in_trial = (inputs[0] @ model.W_ih.T + model.offset_ih + n_t[0]).detach().numpy()
    hidden_sr_trial = model.transfer_func(h_t).detach().numpy()[0]
    syn_eff_trial = r_t.detach().numpy()[0] * u_t.detach().numpy()[0]
    outputs_trial = z_t.detach().numpy()[0]
    targets_trial = targets.detach().numpy()[0]

    # visualize network's response
    if plot:
        # for not, plot injected noise over time as perturbation
        fig = plot_state_traj(perturb=ext_in_trial, h_units=hidden_sr_trial,
                              syn_eff=syn_eff_trial, outputs=outputs_trial,
                              targets=targets_trial, times=times)

    # calculate metrics-of-interest
    # n_dim = est_dimensionality(hidden_sr_trial)
    stats = dict(loss=loss.item())
    # stats = dict(loss=loss.item(), dimensionality=n_dim)

    # package all state variables
    state_vars = (n_t.detach(),
                  h_t.detach(),
                  r_t.detach(),
                  u_t.detach(),
                  z_t.detach())

    return state_vars, stats
