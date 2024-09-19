"""Main development script for project."""

# %%
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn

from utils import get_device


# %% set meta-parameters
device = get_device()
torch.random.manual_seed(1234)  # for reproducibility while troubleshooting


# %% define model
class RNN(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        n_rec_units = 100
        self.noise_std = 0.001
        # self.h_0 = torch.zeros(n_rec_units)
        self.rec_layer = nn.RNN(input_size=n_inputs,
                                hidden_size=n_rec_units,
                                nonlinearity='tanh',
                                bias=False,
                                batch_first=True)
        self.output_layer = nn.Linear(in_features=n_rec_units,
                                      out_features=n_outputs,
                                      bias=False)

    def forward(self, x):
        noise = torch.randn_like(x) * self.noise_std
        h_t, _ = self.rec_layer(x + noise)
        return self.output_layer(h_t)


n_inputs, n_outputs = 1, 1
model = RNN(n_inputs=1, n_outputs=1).to(device)
print(model)


loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# %% create data
# proxy for precision of timing code (increase for more precision)
n_amplitudes = 10
max_perturb = 2.0
min_perturb = -max_perturb

dt = 1e-3  # 1 ms
tstop = 2.  # 2 sec
times = np.arange(-0.1, tstop, dt)
n_samps = len(times)
perturb_dur = 0.02  # 20 ms
perturb_win_mask = np.logical_and(times > -perturb_dur, times < 0)


amplitudes = torch.linspace(min_perturb, max_perturb, n_amplitudes)
data_x = torch.zeros(n_amplitudes, n_samps, n_inputs)
data_x[:, perturb_win_mask, :] = torch.tile(amplitudes[:, np.newaxis, np.newaxis],
                                            (1,
                                            np.sum(perturb_win_mask),
                                            n_inputs))

delays = np.linspace(0.01, tstop - 0.01, n_amplitudes)  # add margins
data_y = torch.zeros(n_amplitudes, n_samps, n_outputs)
for delay in delays:
    delay_mask = times >= delay
    data_y[:, delay_mask, :] = 1.0


# %% define train and test functions that will loop over
# each batch
def train(X, Y, model, loss_fn, optimizer):
    model.train()

    X, Y = X.to(device), Y.to(device)

    # Compute prediction error
    pred = model(X)
    loss = loss_fn(pred, Y)

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
        pred = model(X)
        loss = loss_fn(pred, Y)

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
