"""Optimization functions for ANN models."""

import numpy as np
import matplotlib.pyplot as plt

import torch

from viz import plot_traj


def step_sangers_rule(W, W_mask, inputs, outputs, lr=1e-5):
    '''Online unsupervised learning method implementing Sanger's Rule.'''
    xcov = torch.outer(outputs, inputs)
    autocov = torch.outer(outputs, outputs)
    # mask for non-zero connections
    dW = lr * (xcov - torch.tril(autocov) @ (W * W_mask)) * W_mask
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
            model.W_hh.data += step_sangers_rule(model.W_hh.data,
                                                 model.W_hh_mask,
                                                 h_0[0], h_t[0, 0])

    updated_params = torch.cat((model.W_hh[model.W_hh_mask == 1],
                                model.W_hz.data.flatten()))
    param_dist = (torch.linalg.norm(updated_params - init_params)
                  / torch.linalg.norm(init_params))

    return param_dist.numpy(force=True)


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

    if debug_backprop:
        n_hidden = model.W_hh.data.shape[0]
        W_hh_original = np.full([n_hidden, n_hidden], fill_value=np.nan)
        loss_original = np.nan
        grad_errs = list()

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
        optimizer.step()
        if debug_backprop:
            W_hh_updated = np.array(model.W_hh.data)
            if np.all(np.isfinite(W_hh_original)):
                dloss = np.array(loss.detach()) - loss_original
                dloss_dWhh_true = dloss / dWhh
                mask = np.isfinite(dloss_dWhh_true)
                dloss_dWhh_est = np.array(model.W_hh.grad)
                grad_err = np.linalg.norm(dloss_dWhh_true[mask] - dloss_dWhh_est[mask])
                grad_errs.append(grad_err)

            dWhh = W_hh_updated - W_hh_original
            W_hh_original = W_hh_updated.copy()
            loss_original = np.array(loss.detach())
        optimizer.zero_grad()
        losses.append(loss.item())

    updated_params = torch.cat((model.W_hh[model.W_hh_mask == 1],
                                model.W_hz.data.flatten()))
    # updated_params = updated_params.numpy(force=True)
    # param_dist = scipy.spatial.distance.cosine(init_params, updated_params)
    param_dist = (torch.linalg.norm(updated_params - init_params)
                  / torch.linalg.norm(init_params))

    if debug_backprop:
        return np.mean(losses), param_dist, grad_errs

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
    try:
        print(f"Test loss: {loss.item():>7f}")
    except RuntimeError:
        Warning("Test loss isn't a scalar!")
    return h_t, loss


def set_optimimal_w_out(inputs, targets, times, model, loss_fn, h_0,
                        plot=True):
    dt = times[1] - times[0]
    model.eval()

    with torch.no_grad():

        # Compute prediction error
        outputs, h_t = model(inputs, h_0=h_0, dt=dt)
        loss = loss_fn(outputs[:, times > 0, :], targets[:, times > 0, :])
        transfer_func = torch.nn.Tanh()
        h_transfer = transfer_func(h_t)

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
        # h_argmax = torch.argmax(model.W_hz.abs(), dim=1)

        outputs, h_t = model(inputs, h_0=h_0, dt=dt)
        # plt.figure()
        # plt.plot(times[times > 0], h_transfer[:, h_argmax])
        # plt.ylabel('f(X)')
        # plt.xlabel('time (s)')
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
