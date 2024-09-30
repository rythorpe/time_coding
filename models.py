"""ANN models."""

import numpy as np

import torch
from torch import nn


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
        for param in self.rec_layer.parameters():
            param.detach_()
        self.output_layer = nn.Sequential(
            nn.Linear(in_features=n_rec_units,
                      out_features=n_outputs,
                      bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        noise = torch.randn_like(x) * self.noise_std
        h_t, _ = self.rec_layer(x + noise)
        return self.output_layer(h_t), h_t


class RNN_echostate(nn.Module):
    def __init__(self, n_inputs=1, n_hidden=300, n_outputs=1,
                 silence_feedback=False):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.noise_std = 0.0  # 0.001
        self.tau = 0.01  # 10 ms

        self.W_ih = nn.Parameter(torch.empty(n_hidden, n_inputs),
                                 requires_grad=False)
        self.W_hh = nn.Parameter(torch.empty(n_hidden, n_hidden),
                                 requires_grad=False)
        self.W_hz = nn.Parameter(torch.empty(n_outputs, n_hidden),
                                 requires_grad=True)
        self.W_zh = nn.Parameter(torch.empty(n_hidden, n_outputs),
                                 requires_grad=False)

        # initialize input weights
        w_input_std = 1 / np.sqrt(n_hidden)
        torch.nn.init.normal_(self.W_ih, mean=0.0, std=w_input_std)

        # initialize hidden weights
        gain = 1.5
        prob_c = 0.10
        w_hidden_std = gain / np.sqrt(prob_c * n_hidden)
        torch.nn.init.sparse_(self.W_hh, sparsity=(1 - prob_c),
                              std=w_hidden_std)
        # torch.nn.init.normal_(self.W_hh, mean=0.0, std=w_hidden_std)
        # rand_sources = np.random.choice(n_hidden,
        #                                 size=np.round((1 - prob_c) * n_hidden), # noqa
        #                                 replace=False)
        # rand_targets = np.random.choice(n_hidden,
        #                                 size=np.round((1 - prob_c) * n_hidden), # noqa
        #                                 replace=False)
        # self.W_hh[rand_sources, rand_targets] = 0.0

        # initialize output weights
        w_output_std = 1 / np.sqrt(n_hidden)
        torch.nn.init.normal_(self.W_hz, mean=0.0, std=w_output_std)

        # initialize feedback (echo) weights
        if silence_feedback:
            torch.nn.init.zeros_(self.W_zh)
        else:
            torch.nn.init.uniform_(self.W_zh, a=-1.0, b=1.0)

    def forward(self, x, h_0=None, dt=0.001):
        # assuming batches x time x n_inputs
        batch_size, seq_len, _ = x.size()

        if h_0 is None:
            h_0 = (torch.rand(self.n_hidden) * 2) - 1  # uniform in (-1, 1)
            h_0 = torch.tile(h_0, (batch_size, 1))  # replicate for each batch
        h = torch.zeros(batch_size, seq_len, self.n_hidden)
        z = torch.zeros(batch_size, seq_len, self.n_outputs)

        for batch_idx in range(batch_size):
            # initialize hidden and output states
            # each batch can theoretically have a different start point
            h_t_minus_1 = h_0[batch_idx, :]
            h_transfer = torch.tanh(h_0[batch_idx, :])
            z_t_minus_1 = h_transfer @ self.W_hz.T
            # begin integration over time
            for t in range(seq_len):
                noise = torch.randn(self.n_hidden) * self.noise_std
                dhdt = (-h_t_minus_1 + h_transfer @ self.W_hh.T
                        + x[batch_idx, t] @ self.W_ih.T
                        + z_t_minus_1 @ self.W_zh.T
                        + noise) / self.tau
                h_t = h_t_minus_1 + dhdt * dt
                h[batch_idx, t, :] = h_t
                h_t_minus_1 = h_t
                # compute h output here so that it can be passed to both
                # z (output unit) and itself on the next time step
                h_transfer = torch.tanh(h_t)
                z_t = h_transfer @ self.W_hz.T
                z[batch_idx, t, :] = z_t
                z_t_minus_1 = z_t
        return z, h
