"""Main development script for project."""

# %%
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn

from utils import get_device
from models import RNN
from viz import plot_inputs_outputs


# %% set meta-parameters
device = get_device()
torch.random.manual_seed(1234)  # for reproducibility while troubleshooting


# %% instantiate model, loss function, and optimizer
n_inputs, n_outputs = 1, 1
model = RNN(n_inputs=1, n_outputs=1).to(device)
print(model)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)


# %% create data
# proxy for precision of timing code (increase for more precision)
n_amplitudes = 5
min_perturb, max_perturb = -2.0, 2.0

dt = 1e-3  # 1 ms
tstop = 1.  # 1 sec
times = np.arange(-0.1, tstop, dt)
n_samps = len(times)
perturb_dur = 0.025  # 25 ms
perturb_win_mask = np.logical_and(times > -perturb_dur, times < 0)


amplitudes = torch.linspace(min_perturb, max_perturb, n_amplitudes)
data_x = torch.zeros((n_amplitudes, n_samps, n_inputs))
data_x[:, perturb_win_mask, :] = torch.tile(
    amplitudes[:, np.newaxis, np.newaxis],
    (1, np.sum(perturb_win_mask), n_inputs)
)

delays = np.linspace(0.1, tstop - 0.1, n_amplitudes)  # add margins
data_y = torch.zeros((n_amplitudes, n_samps, n_outputs))
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
def train(X, Y, model, loss_fn, optimizer):
    model.train()

    X, Y = X.to(device), Y.to(device)

    # Compute prediction error
    pred, h_t = model(X)
    loss = loss_fn(pred[:, times > 0, :],
                   Y[:, times > 0, :])

    # Backpropagation
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print(f"Training loss: {loss.item():>7f}")
    return loss.item()


def test(X, Y, model, loss_fn):
    model.eval()

    with torch.no_grad():
        X, Y = X.to(device), Y.to(device)

        # Compute prediction error
        pred, h_t = model(X)
        loss = loss_fn(pred, Y)

        input = X.cpu()
        pred = pred.cpu()
        rec_traj = h_t.cpu()

        fig = plot_inputs_outputs(input, pred, times, rec_traj=rec_traj)
        fig.show()

    print(f"Test loss: {loss.item():>7f}")


# %% train and test model over a few epochs
epochs = 100
loss_per_step = list()
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    loss = train(data_x, data_y, model, loss_fn, optimizer)
    loss_per_step.append(loss)
    # test(test_dataloader, model, loss_fn)
print("Done!")

# %%
test(data_x, data_y, model, loss_fn)

# %%
