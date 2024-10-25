"""Optimization functions for ANN models."""

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.optim import Optimizer

from viz import plot_traj


def diff_loss(output, target):
    epsilon = target - output
    print(epsilon)
    return epsilon


class RLS_opt():
    def __init__(self, params, n_params, alpha=0.5):
        # super().__init__()
        # NB: assumes that we are only optimizing output weights (W_hz)
        self.params = params
        self.P = torch.eye(n_params) / alpha

    def step(self, h_response):
        for h_dim in range(h_response.shape[0]):
            h_dim_response = h_response[[h_dim]]
            self.P = self.P - ((self.P @ h_dim_response.T @ h_dim_response @ self.P) /
                               (1 + h_dim_response @ self.P @ h_dim_response.T))
            for W in self.params:
                if W.requires_grad:
                    W = W + W.grad @ self.P @ h_dim_response


def RLS_fit(P, W, h_response, err):
    P = P - ((P @ h_response.T @ h_response @ P) /
             (1 + h_response @ P @ h_response.T))
    W = W + err * P @ h_response


def train(inputs, targets, times, model, loss_fn, optimizer, h_0,
          debug_backprop=False):
    dt = times[1] - times[0]
    n_times = len(times)
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

    return np.mean(losses)


def test(inputs, targets, times, model, loss_fn, h_0):
    dt = times[1] - times[0]
    model.eval()

    with torch.no_grad():

        # Compute prediction error
        outputs, h_t = model(inputs, h_0=h_0, dt=dt)
        loss = loss_fn(outputs[:, times > 0, :], targets[:, times > 0, :])

    h_t_batch = h_t.cpu().squeeze()
    outputs_batch = outputs.cpu().squeeze()
    targets_batch = targets.cpu().squeeze()

    fig = plot_traj(h_units=h_t_batch, outputs=outputs_batch,
                    targets=targets_batch, times=times)
    fig.show()
    print(f"Test loss: {loss.item():>7f}")


def set_optimimal_w_out(inputs, targets, times, model, loss_fn, h_0):
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
    fig = plot_traj(h_units=h_t_batch, outputs=outputs_batch,
                    targets=targets_batch, times=times)
    fig.show()
