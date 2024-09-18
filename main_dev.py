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
        self.h_0 = torch.zeros(n_rec_units)
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
        h_t, _ = self.rec_layer(x + noise, self.h_0)
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


amplitudes = np.linspace(min_perturb, max_perturb, n_amplitudes)
data_x = torch.zeros(n_amplitudes, n_samps, n_inputs)
data_x[:, perturb_win_mask, :] = amplitudes

delays = np.linspace(0.01, tstop - 0.01, n_amplitudes)
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

    print(f"loss: {loss.item():>7f}")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# %% train and test model over a few epochs
epochs = 30
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(data_x, data_y, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")