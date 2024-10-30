"""Optimization functions for ANN models."""

import numpy as np
import matplotlib.pyplot as plt

import torch

from viz import plot_traj


def step_sangers_rule(W, inputs, outputs, lr=1e-3):
    '''Online unsupervised learning method implementing Sanger's Rule.'''
    xcov = torch.outer(outputs, inputs)
    autocov = torch.outer(outputs, outputs)
    dW = lr * (xcov - torch.tril(autocov) @ W)
    return dW


def pre_train(inputs, times, model, h_0):
    dt = times[1] - times[0]
    n_times = len(times)
    init_params = torch.cat((model.W_hh[model.W_hh_mask == 1],
                             model.W_hz.data.flatten()))
    model.train()

    # run model without storing gradients until t=0
    with torch.no_grad():
        outputs, h_t = model(inputs[:, times <= 0, :], h_0=h_0, dt=dt)

        # now, train using at each time point using Sanger's Rule
        step_size = 1
        t_0_idx = np.nonzero(times > 0)[0][0]
        for t_idx in np.arange(t_0_idx + step_size, n_times + step_size,
                               step_size):
            # compute prediction error
            t_minus_1_idx = t_idx - step_size
            h_0 = h_t[:, -1, :].detach()
            outputs, h_t = model(inputs[:, t_minus_1_idx:t_idx, :], h_0=h_0,
                                 dt=dt)
            model.W_hh.data += step_sangers_rule(h_0, h_t)

    updated_params = torch.cat((model.W_hh[model.W_hh_mask == 1],
                                model.W_hz.data.flatten()))
    param_dist = (torch.linalg.norm(updated_params - init_params)
                  / torch.linalg.norm(init_params))

    return param_dist


def train(inputs, targets, times, model, loss_fn, optimizer, h_0,
          debug_backprop=False):
    dt = times[1] - times[0]
    n_times = len(times)
    init_params = torch.cat((model.W_hh[model.W_hh_mask == 1],
                             model.W_hz.data.flatten()))
    # init_params = init_params.numpy(force=True)
    model.train()

    # run model without storing gradients until t=0
    with torch.no_grad():
        outputs, h_t = model(inputs[:, times <= 0, :], h_0=h_0, dt=dt)

    # if debug_backprop:
    #     dWhh_dloss_true = torch.empty(n_hidden, n_hidden,
    #                                   requires_grad=False)

    # now, train using FORCE
    step_size = 1
    losses = list()
    t_0_idx = np.nonzero(times > 0)[0][0]
    for t_idx in np.arange(t_0_idx + step_size, n_times + step_size,
                           step_size):
        # compute prediction error
        t_minus_1_idx = t_idx - step_size
        h_0 = h_t[:, -1, :].detach()
        outputs, h_t = model(inputs[:, t_minus_1_idx:t_idx, :], h_0=h_0, dt=dt)
        # loss at t - delta_t
        loss = loss_fn(outputs[:, 0, :], targets[:, t_minus_1_idx, :])
        # backpropagation
        loss.backward()

        # if debug_backprop:
        #     dWhh_dloss_true = model.W_hh[:, :].copy().flatten()

        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())

    updated_params = torch.cat((model.W_hh[model.W_hh_mask == 1],
                                model.W_hz.data.flatten()))
    # updated_params = updated_params.numpy(force=True)
    # param_dist = scipy.spatial.distance.cosine(init_params, updated_params)
    param_dist = (torch.linalg.norm(updated_params - init_params)
                  / torch.linalg.norm(init_params))

    return np.mean(losses), param_dist


def test(inputs, targets, times, model, loss_fn, h_0, plot=True):
    dt = times[1] - times[0]
    model.eval()

    with torch.no_grad():

        # Compute prediction error
        outputs, h_t = model(inputs, h_0=h_0, dt=dt)
        loss = loss_fn(outputs[:, times > 0, :], targets[:, times > 0, :])

    h_t_batch = h_t.cpu().squeeze()
    outputs_batch = outputs.cpu().squeeze()
    targets_batch = targets.cpu().squeeze()

    if plot:
        fig = plot_traj(h_units=h_t_batch, outputs=outputs_batch,
                        targets=targets_batch, times=times)
        fig.show()
    print(f"Test loss: {loss.item():>7f}")
    return h_t, loss


def set_optimimal_w_out(inputs, targets, times, model, loss_fn, h_0,
                        plot=True):
    dt = times[1] - times[0]
    model.eval()

    with torch.no_grad():

        # Compute prediction error
        outputs, h_t = model(inputs, h_0=h_0, dt=dt)
        loss = loss_fn(outputs[:, times > 0, :], targets[:, times > 0, :])
        activation = torch.nn.Tanh()
        h_transfer = activation(h_t)

        h_t = h_t.cpu()
        outputs = outputs.cpu()
        # assuming batch size of 1, select first and only batch
        h_transfer = h_transfer.cpu()[0, times > 0, :]

        cov = h_transfer.T @ h_transfer
        # once again, assuming batch size of 1
        targets_ = targets[0, times > 0, :]  # column vectors
        # W_hz_ = cov.inverse() @ (h_transfer.T @ targets_)
        W_hz_ = torch.linalg.solve(cov, (h_transfer.T @ targets_))
        model.W_hz[:] = W_hz_.T
        h_argmax = torch.argmax(model.W_hz.abs(), dim=1)

        outputs, h_t = model(inputs, h_0=h_0, dt=dt)
        plt.figure()
        plt.plot(times[times > 0], h_transfer[:, h_argmax])
        plt.ylabel('f(X)')
        plt.xlabel('time (s)')
        loss = loss_fn(outputs[:, times > 0, :], targets[:, times > 0, :])
        print(f"Min. loss: {loss.item():>7f}")

    h_t_batch = h_t.cpu().squeeze()
    outputs_batch = outputs.cpu().squeeze()
    targets_batch = targets.cpu().squeeze()
    if plot:
        fig = plot_traj(h_units=h_t_batch, outputs=outputs_batch,
                        targets=targets_batch, times=times)
        fig.show()
    return outputs
