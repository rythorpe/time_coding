"""Main development script for project."""

# %%
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn

from utils import get_device, gaussian
from models import RNN
from opt import diff_loss, RLS_opt
from viz import plot_inputs_outputs


# %% set meta-parameters
# device = get_device()
device = 'cpu'
torch.random.manual_seed(1234)  # for reproducibility while troubleshooting


# %% instantiate model, loss function, and optimizer
n_inputs, n_hidden, n_outputs = 1, 300, 1
model = RNN(n_inputs=n_inputs, n_hidden=n_hidden,
            n_outputs=n_outputs, echo_state=False)
model.to(device)
print(model)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


# %% create data
# proxy for precision of timing code (increase for more precision)
n_amplitudes = 3
min_perturb, max_perturb = -2.0, 2.0

dt = 1e-3  # 1 ms
tstop = 1.  # 1 sec
times = np.arange(-0.1, tstop, dt)
n_times = len(times)
perturb_dur = 0.05  # 50 ms
perturb_win_mask = np.logical_and(times > -perturb_dur, times < 0)


amplitudes = torch.linspace(min_perturb, max_perturb, n_amplitudes)
data_x = torch.zeros((n_amplitudes, n_times, n_inputs))
data_x[:, perturb_win_mask, :] = torch.tile(
    amplitudes[:, np.newaxis, np.newaxis],
    (1, np.sum(perturb_win_mask), n_inputs)
)

delays = np.linspace(0.1, tstop - 0.1, n_amplitudes)  # add margins
# delays = [0.5]
width = 0.02  # 20 ms
data_y = torch.zeros((n_amplitudes, n_times, n_inputs))
for output_idx, center in enumerate(delays):
    data_y[output_idx, :, 0] = torch.tensor(gaussian(times, center, width))
    # data_y[output_idx, :, 0] = torch.tensor(np.sin(times * 2 * np.pi * 2))


# %% define train and test functions that will loop over
# each batch
def train(inputs, targets, times, model, loss_fn, optimizer, h_0=None,
          debug_backprop=False):
    model.train()
    inputs, targets = inputs.to(device), targets.to(device)

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

    return np.sum(losses)


def test(inputs, targets, times, model, loss_fn, h_0=None):
    model.eval()

    with torch.no_grad():
        inputs, targets = inputs.to(device), targets.to(device)

        # Compute prediction error
        outputs, h_t = model(inputs, h_0=h_0, dt=dt)
        loss = loss_fn(outputs[:, times > 0, :], targets[:, times > 0, :])

        inputs = inputs.cpu()
        outputs = outputs.cpu()
        h_t = h_t.cpu()

    fig = plot_inputs_outputs(inputs, outputs, times, rec_traj=h_t,
                              targets=targets)
    fig.show()
    print(f"Test loss: {loss.item():>7f}")


def set_optimimal_w_out(inputs, targets, times, model, loss_fn, h_0):
    model.eval()

    with torch.no_grad():
        inputs, targets = inputs.to(device), targets.to(device)

        # Compute prediction error
        outputs, h_t = model(inputs, h_0=h_0, dt=dt)
        loss = loss_fn(outputs[:, times > 0, :], targets[:, times > 0, :])
        print(f"Initial loss: {loss.item():>7f}")
        activation = torch.nn.Tanh()
        h_transfer = activation(h_t)

        inputs = inputs.cpu()
        outputs = outputs.cpu()
        h_t = h_t.cpu()
        # assuming input/output size of 1, select first and only batch
        h_transfer = h_transfer.cpu()[0, times > 0, :]

        cov = h_transfer.T @ h_transfer
        targets_ = targets[0, times > 0, 0:1]  # column vector
        # W_hz_ = cov.inverse() @ (h_transfer.T @ targets_)
        W_hz_ = torch.linalg.solve(cov, (h_transfer.T @ targets_))
        model.W_hz[:] = W_hz_.T
        h_argmax = torch.argmax(model.W_hz.abs())

        outputs, h_t = model(inputs, h_0=h_0, dt=dt)
        plt.figure()
        plt.plot(times[times > 0], h_transfer[:, h_argmax])
        plt.ylabel('X (hidden)')
        plt.xlabel('time (s)')
        loss = loss_fn(outputs[:, times > 0, :], targets[:, times > 0, :])
        print(f"Final loss: {loss.item():>7f}")

    fig = plot_inputs_outputs(inputs, outputs, times, rec_traj=h_t,
                              targets=targets)
    axes = fig.get_axes()
    axes[1].set_ylabel('X')
    fig.show()


# fix initial conditions for training and testing
h_0 = (torch.rand(n_hidden) * 2) - 1  # uniform in (-1, 1)
h_0 = torch.tile(h_0, (n_amplitudes, 1))  # replicate for each batch
h_0 = h_0.to(device)

# plot model output before training
test(data_x, data_y, times, model, loss_fn, h_0=h_0)

# %% train and test model over a few epochs
n_iter = 50
loss_per_iter = list()
for t in range(n_iter):
    print(f"Iteration {t+1}\n-------------------------------")
    loss = train(data_x, data_y, times, model, loss_fn, optimizer, h_0=h_0)
    loss_per_iter.append(loss)
print("Done!")

plt.figure()
plt.plot(loss_per_iter)
plt.xlabel('iteration')
plt.ylabel('loss')

# %%
# plot model output after training
test(data_x, data_y, times, model, loss_fn, h_0=h_0)
# plot optimal model output given hidden unit responses
set_optimimal_w_out(data_x, data_y, times, model, loss_fn, h_0=h_0)

# %%
