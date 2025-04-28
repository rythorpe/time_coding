"""Training and test functions for RNN model."""

import numpy as np
from scipy.spatial import distance
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
    init_params = model.W_hz.data.flatten()
    # init_params = init_params.numpy(force=True)
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

        # print(model.presyn_scaling.grad)

        # if debug_backprop:
        #     dWhh_dloss_true = model.W_hh[:, :].copy().flatten()

        optimizer.step()
        optimizer.zero_grad()

        # reset presyn_scaling vector
        model.W_hh *= model.presyn_scaling.detach()
        torch.nn.init.ones_(model.presyn_scaling)

        losses.append(loss.item())

    updated_params = model.W_hz.data.flatten()
    # updated_params = updated_params.numpy(force=True)
    # param_dist = scipy.spatial.distance.cosine(init_params, updated_params)
    param_dist = (torch.linalg.norm(updated_params - init_params)
                  / torch.linalg.norm(init_params))

    return np.mean(losses), param_dist


def train_bptt_sparse(inputs, targets, times, model, loss_fn, optimizer,
                      h_0, r_0, u_0, p_backprop=0.2):
    dt = times[1] - times[0]
    n_times = len(times)
    # init_params = torch.cat([par.detach().flatten()
    #                          for par in model.parameters()])
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

        # for t_idx in range(len(h_t_all)):
        #     h_t_all[-t_idx - 1].backward(h_t_all[-t_idx].grad, retain_graph=True)

    optimizer.step()
    # reset presyn_scaling vector
    # model.W_hh *= model.presyn_scaling.detach()
    # torch.nn.init.ones_(model.presyn_scaling)
    optimizer.zero_grad()

    # updated_params = torch.cat([par.detach().flatten()
    #                             for par in model.parameters()])
    # param_dist = distance.cosine(init_params.numpy(force=True),
    #                              updated_params.numpy(force=True))
    param_dist = None

    return np.sum(losses), param_dist


def train_bptt(inputs, targets, times, model, loss_fn, optimizer,
               h_0, r_0, u_0):
    dt = times[1] - times[0]
    n_times = len(times)
    init_params = model.W_hz.data.flatten()
    # init_params = init_params.numpy(force=True)
    model.train()

    h_t, r_t, u_t, z_t = model(inputs, h_0=h_0, r_0=r_0, u_0=u_0, dt=dt)
    loss = loss_fn(z_t[:, times > 0, :], targets[:, times > 0, :])
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()

    updated_params = model.W_hz.data.flatten()
    # updated_params = updated_params.numpy(force=True)
    # param_dist = scipy.spatial.distance.cosine(init_params, updated_params)
    param_dist = (torch.linalg.norm(updated_params - init_params)
                  / torch.linalg.norm(init_params))

    return loss.item(), param_dist


def set_optimimal_w_out(inputs, targets, times, model, loss_fn, h_0,
                        plot=True):
    dt = times[1] - times[0]
    model.eval()

    with torch.no_grad():

        # Compute prediction error
        h_t, r_t, u_t, z_t = model(inputs, h_0=h_0, dt=dt)
        loss = loss_fn(z_t[:, times > 0, :], targets[:, times > 0, :])
        h_transfer = torch.tanh(h_t)

        # assuming batch size of 1, select first and only batch
        h_transfer = h_transfer.cpu()[0, times > 0, :]

        cov = h_transfer.T @ h_transfer
        # once again, assuming batch size of 1
        targets_ = targets[0, times > 0, :]  # column vectors
        # W_hz_ = cov.inverse() @ (h_transfer.T @ targets_)
        W_hz_ = torch.linalg.solve(cov, (h_transfer.T @ targets_))
        model.W_hz[:] = W_hz_.T
        # h_argmax = torch.argmax(model.W_hz.abs(), dim=1)

        h_t, r_t, u_t, z_t = model(inputs, h_0=h_0, dt=dt)
        # plt.figure()
        # plt.plot(times[times > 0], h_transfer[:, h_argmax])
        # plt.ylabel('f(X)')
        # plt.xlabel('time (s)')
        loss = loss_fn(z_t[:, times > 0, :], targets[:, times > 0, :])
        print(f"Min. loss: {loss.item():>7f}")

    # select first batch if more than one exists
    hidden_batch = torch.tanh(h_t).cpu()[0]
    outputs_batch = z_t.cpu()[0]
    targets_batch = targets.cpu()[0]

    if plot:
        fig = plot_state_traj(h_units=hidden_batch, outputs=outputs_batch,
                              targets=targets_batch, times=times)
        fig.show()
    return z_t.cpu()


def test_and_get_stats(inputs, targets, times, model, loss_fn, h_0, r_0, u_0,
                       plot=True):
    dt = times[1] - times[0]
    model.eval()

    with torch.no_grad():
        # simulate and calculate total output error
        h_t, r_t, u_t, z_t = model(inputs, h_0=h_0, r_0=r_0, u_0=u_0, dt=dt)
        loss = loss_fn(z_t[:, times > 0, :], targets[:, times > 0, :])
    
    try:
        print(f"Test loss: {loss.item():>7f}")
    except RuntimeError:
        Warning("Test loss isn't a scalar!")

    # select first batch if more than one exists
    hidden_batch = torch.tanh(h_t).cpu()[0]
    outputs_batch = z_t.cpu()[0]
    targets_batch = targets.cpu()[0]

    # visualize network's response
    if plot:
        fig = plot_state_traj(h_units=hidden_batch, outputs=outputs_batch,
                              targets=targets_batch, times=times)
        fig.show()

    # calculate metrics-of-interest
    n_dim = est_dimensionality(hidden_batch)
    stats = dict(loss=loss, dimensionality=n_dim)
    
    return torch.tanh(h_t).cpu(), z_t.cpu(), stats
