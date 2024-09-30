"""Main development script for project."""

# %%
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn

from utils import get_device
from models import RNN_echostate
from viz import plot_inputs_outputs


# %% set meta-parameters
device = get_device()
torch.random.manual_seed(1234)  # for reproducibility while troubleshooting


# %% instantiate model, loss function, and optimizer
n_inputs, n_hidden, n_outputs = 1, 1000, 1
model = RNN_echostate(n_inputs=1, n_hidden=n_hidden, n_outputs=1,
                      silence_feedback=False).to(device)
print(model)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


# %% create data
# proxy for precision of timing code (increase for more precision)
n_amplitudes = 1
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

# delays = np.linspace(0.1, tstop - 0.1, n_amplitudes)  # add margins
delays = [0.5]
data_y = torch.zeros((n_amplitudes, n_times, n_outputs))
k_w = 50  # numel of a given side of the kernel
gaussian_kernel = np.exp(np.arange(-k_w * dt, k_w * dt, dt) ** 2 /
                         (-2 * (dt * 10) ** 2))
gaussian_kernel /= np.sum(gaussian_kernel)  # normalize
for delay_idx, delay in enumerate(delays):
    delay_mask = times >= delay
    data_y[delay_idx, delay_mask, :] = 1.0
    for output_idx in range(n_outputs):
        unsmoothed = np.concatenate([np.zeros(k_w),
                                     data_y[delay_idx, :, output_idx].numpy(),
                                     np.ones(k_w)])
        smoothed = np.convolve(
            unsmoothed,
            gaussian_kernel,
            mode='same'
        )
        data_y[delay_idx, :, output_idx] = torch.Tensor(smoothed[k_w:-k_w])

fig = plot_inputs_outputs(data_x, data_y, times)
fig.show()


# %% define train and test functions that will loop over
# each batch
def train(X, Y, model, loss_fn, optimizer, h_0=None):
    model.train()
    X, Y = X.to(device), Y.to(device)

    # run model without storing gradients until t=0
    with torch.no_grad():
        pred, h_t = model(X[:, times <= 0, :], h_0=h_0, dt=dt)

    # now, train using FORCE
    force_step = 2
    losses = list()
    t_0_idx = np.nonzero(times > 0)[0][0]
    for t_idx in np.arange(t_0_idx + force_step, n_times + force_step,
                           force_step):
        # Compute prediction error
        t_minus_1_idx = t_idx - force_step
        h_0 = h_t[:, -1, :].detach()
        pred, h_t = model(X[:, t_minus_1_idx:t_idx, :], h_0=h_0, dt=dt)
        # loss at t - delta_t
        loss = loss_fn(pred[:, 0, :], Y[:, t_minus_1_idx, :])
        # backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())

    return losses


def test(X, Y, model, loss_fn, h_0=None):
    model.eval()

    with torch.no_grad():
        X, Y = X.to(device), Y.to(device)

        # Compute prediction error
        pred, h_t = model(X, h_0=h_0, dt=dt)
        loss = loss_fn(pred, Y)

        input = X.cpu()
        pred = pred.cpu()
        h_t = h_t.cpu()

        fig = plot_inputs_outputs(input, pred, times, rec_traj=h_t)
        fig.show()

    print(f"Test loss: {loss.item():>7f}")


# fix initial conditions for training and testing
h_0 = (torch.rand(n_hidden) * 2) - 1  # uniform in (-1, 1)
h_0 = torch.tile(h_0, (n_amplitudes, 1))  # replicate for each batch
h_0.to(device)

# %% train and test model over a few epochs
n_iter = 5
loss_per_iter = list()
for t in range(n_iter):
    print(f"Iteration {t+1}\n-------------------------------")
    loss = train(data_x, data_y, model, loss_fn, optimizer, h_0=h_0)
    loss_per_iter.extend(loss)
    # test(test_dataloader, model, loss_fn)
print("Done!")

plt.figure()
plt.plot(loss_per_iter)

# %%
test(data_x, data_y, model, loss_fn, h_0=h_0)

# %%
