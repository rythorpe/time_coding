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
                                           h_0=h_0, r_0=r_0, u_0=u_0, dt=dt)

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
                h_0=h_0, r_0=r_0, u_0=u_0, dt=dt)
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
               h_0, r_0, u_0, dt):
    model.train()

    init_params = [param.detach().numpy() for param in model.parameters()
                   if param.requires_grad]
    W_hh_orig = model.W_hh.data.detach().clone()

    state_vars = model(inputs, h_0=h_0, r_0=r_0, u_0=u_0, dt=dt)
    z_t = state_vars[3]
    loss = loss_fn(z_t[:, times > 0, :], targets[:, times > 0, :])
    loss.backward()

    optimizer.step()
    # scale W_hh sparsely and reset presyn_scaling vector
    # model.W_hh[:, :50] *= model.presyn_scaling.detach()[:50]
    # torch.nn.init.ones_(model.presyn_scaling)
    optimizer.zero_grad()

    # enforce constraints on learned recurrent weights
    with torch.no_grad():
        # enforce Dale's Law by preventing E/I units from changing valance
        e_weights = model.W_hh[model.W_hh_mask == 1]
        i_weights = model.W_hh[model.W_hh_mask == -1]
        e_weights_clipped = torch.nn.functional.relu(e_weights)
        i_weights_clipped = -1 * torch.nn.functional.relu(-1 * i_weights)
        model.W_hh[model.W_hh_mask == 1] = e_weights_clipped
        model.W_hh[model.W_hh_mask == -1] = i_weights_clipped
        # enforce sparse connectivity
        model.W_hh[model.W_hh_mask == 0] = 0.0
        # now, stabilize learning by setting majority of non-zero connections
        # back to previous state to enforce sparse updates
        source_idxs, target_idxs = torch.nonzero(model.W_hh_mask, as_tuple=True)
        n_conn_total = len(source_idxs)
        n_conn_static = int(np.round(0.5 * n_conn_total))
        rand_idxs_source = torch.randperm(n_conn_total)[:n_conn_static]
        rand_idxs_targ = torch.randperm(n_conn_total)[:n_conn_static]

        params_reset = W_hh_orig[source_idxs[rand_idxs_source],
                                 target_idxs[rand_idxs_targ]]

        model.W_hh[source_idxs[rand_idxs_source],
                   target_idxs[rand_idxs_targ]] = params_reset

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


def sim_batch(inputs, model, h_0, r_0, u_0, dt):
    model.eval()
    with torch.no_grad():
        # simulate network
        h_t, r_t, u_t, z_t = model(inputs, h_0=h_0, r_0=r_0, u_0=u_0, dt=dt)
    return h_t, r_t, u_t, z_t


def test_and_get_stats(inputs, targets, times, model, loss_fn, h_0, r_0, u_0,
                       dt, plot=True, inputs_to_plot=None):
    model.eval()

    with torch.no_grad():
        # simulate and calculate total output error
        h_t, r_t, u_t, z_t = model(inputs, h_0=h_0, r_0=r_0, u_0=u_0, dt=dt)
        loss = loss_fn(z_t[:, times > 0, :], targets[:, times > 0, :])

    try:
        print(f"Test loss: {loss.item():>7f}")
    except RuntimeError:
        Warning("Test loss isn't a scalar!")

    # select first batch trial to visualize single-trial trajectories
    if inputs_to_plot is None:
        ext_in_trial = (inputs[0] + model.offset_ih).detach().numpy()
    else:
        ext_in_trial = inputs_to_plot.detach().numpy()[0]
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
        axes = fig.get_axes()
        axes[0].set_ylim([-1.2, 1.2])

    # calculate metrics-of-interest
    # n_dim = est_dimensionality(hidden_sr_trial)
    stats = dict(loss=loss.item())
    # stats = dict(loss=loss.item(), dimensionality=n_dim)

    # package all state variables
    state_vars = (h_t.detach(),
                  r_t.detach(),
                  u_t.detach(),
                  z_t.detach())

    return state_vars, stats
