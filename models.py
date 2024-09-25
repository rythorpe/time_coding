"""ANN models."""

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
    def __init__(self, n_inputs=1, n_rec_units=100, n_outputs=1):
        super().__init__()
        self.n_rec_units = n_rec_units
        self.n_outputs = n_outputs
        self.noise_std = 0.001
        self.tau = 0.01  # 10 ms
        # self.h_0 = torch.zeros(n_rec_units)
        W_ih = torch.empty(n_inputs,n_rec_units)
        torch.nn.init.normal_(W_ih)
        self.W_ih = nn.Parameter(W_ih, requires_grad=False)
        self.W_hh = nn.Parameter(, requires_grad=False)
        self.W_hz = nn.Parameter(, requires_grad=True)

    def forward(self, x, h_0=None, dt=0.001):
        # transpose so it is time x batches x n_inputs
        x = x.transpose(0, 1)
        seq_len, batch_size, _ = x.size()
        if h_0 is None:
            h_0 = torch.zeros(batch_size, self.n_rec_units)
        h_t_minus_1 = h_0
        h_transfer = torch.tanh(h_0)
        h = torch.zeros(batch_size, seq_len, self.n_rec_units)
        z = torch.zeros(batch_size, seq_len, self.n_outputs)
        for t in range(seq_len):
            noise = torch.randn(self.n_rec_units) * self.noise_std
            # compute h output here so that it can be passed to both
            # z (output unit) and itself on the next time step
            h_transfer = torch.tanh(h_t_minus_1)

            z_t = h_t_minus_1 @ self.W_hz.T
            dhdt = (-h_t_minus_1 + h_transfer @ self.W_hh.T
                    + x[t] @ self.W_ih.T
                    + z_t_minus_1 @ self.W_zh.T
                    + noise) / self.tau
            h_t = h_t_minus_1 + dhdt * dt

            h[:, t, :] = h_t
            h_t_minus_1 = h_t

            z[:, t, :] = z_t
            z_t_minus_1 = z_t
        return z, h
