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
        return self.output_layer(h_t)
