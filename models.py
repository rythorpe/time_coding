"""ANN models."""

import numpy as np

import torch
from torch import nn


class RNN(nn.Module):
    def __init__(self, n_inputs=1, n_hidden=300, n_outputs=1,
                 echo_state=True):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.noise_std = 0.0  # 0.001
        self.tau = 0.01  # 10 ms
        self.tau_depr = 0.2  # 200 ms; taken from Mongillo et al. Science 2008
        self.tau_facil = 1.5  # 1.5 s
        gain = 1.8
        prob_c = 0.3

        self.p_rel = nn.Parameter(torch.empty(n_hidden), requires_grad=False)
        self.tau_depr = nn.Parameter(torch.empty(n_hidden), requires_grad=False)
        self.W_ih = nn.Parameter(torch.empty(n_hidden, n_inputs),
                                 requires_grad=False)
        self.W_hh = nn.Parameter(torch.empty(n_hidden, n_hidden),
                                 requires_grad=True)
        self.W_hz = nn.Parameter(torch.empty(n_outputs, n_hidden),
                                 requires_grad=True)
        self.W_zh = nn.Parameter(torch.empty(n_hidden, n_outputs),
                                 requires_grad=False)
        
        # initialize release probabilities
        # Bounds taken from Tsodyks & Markram PNAS 1997
        torch.nn.init.uniform_(self.p_rel, a=0.1, b=0.95)
        torch.nn.init.uniform_(self.tau_depr, a=0.01, b=0.2)

        # initialize input weights
        w_input_std = 1 / np.sqrt(n_hidden)
        torch.nn.init.normal_(self.W_ih, mean=0.0, std=w_input_std)

        # initialize hidden weights
        w_hidden_std = gain / np.sqrt(prob_c * n_hidden)
        # torch.nn.init.sparse_(self.W_hh, sparsity=(1 - prob_c),
        #                       std=w_hidden_std)
        torch.nn.init.normal_(self.W_hh, mean=0.0, std=w_hidden_std)
        # create mask for non-zero connections; tuning weights of
        # zeroed connections won't effect model dynamics
        n_conns_possible = n_hidden ** 2
        n_conns_chosen = int(np.round(prob_c * n_hidden ** 2))
        rand_conns = np.random.choice(n_conns_possible, size=n_conns_chosen,
                                      replace=False)
        self.W_hh_mask = torch.zeros(n_conns_possible, requires_grad=False)
        self.W_hh_mask[rand_conns] = 1
        self.W_hh_mask = torch.reshape(self.W_hh_mask, (n_hidden, n_hidden))

        # initialize output weights
        w_output_std = 1 / np.sqrt(n_hidden)
        torch.nn.init.normal_(self.W_hz, mean=0.0, std=w_output_std)

        # initialize feedback (echo) weights
        if echo_state:
            torch.nn.init.uniform_(self.W_zh, a=-1.0, b=1.0)
        else:
            torch.nn.init.zeros_(self.W_zh)

        # create registered buffers (i.e., fancy attributes that need to live
        # on the same device as self
        # self.register_buffer('h_0', torch.zeros(self.n_hidden))
        self.register_buffer('noise', torch.zeros(self.n_hidden))

    def forward(self, x, h_0, dt=0.001):
        # assuming batches x time x n_inputs
        batch_size, seq_len, _ = x.size()

        u = torch.zeros(batch_size, seq_len, self.n_hidden)
        h = torch.zeros(batch_size, seq_len, self.n_hidden)
        z = torch.zeros(batch_size, seq_len, self.n_outputs)

        # NB: doesn't work on CUDA; due to FORCE training, h_0 is updated
        # regularly in time and therefore lives on the CPU
        for batch_idx in range(batch_size):
            # initialize hidden and output states
            # each batch can theoretically have a different start point
            u_t_minus_1 = torch.tensor(self.p_rel)
            h_t_minus_1 = h_0[batch_idx, :]
            h_transfer = torch.tanh(h_0[batch_idx, :])
            z_t_minus_1 = h_transfer @ self.W_hz.T
            # begin integration over time
            for t in range(seq_len):
                self.noise.normal_(0, self.noise_std)
                # pre-synaptic plasticity
                dudt = ((self.p_rel - u_t_minus_1) / self.tau_depr
                        - self.p_rel * u_t_minus_1 * h_transfer)
                # dudt = ((self.p_rel - u_t_minus_1) / self.tau_facil
                #         + self.p_rel * (1 - u_t_minus_1) * h_transfer)
                u_t = u_t_minus_1 + dudt * dt
                u[batch_idx, t, :] = u_t
                # post-synaptic integration
                dhdt = (-h_t_minus_1
                        # mask to enforces static, sparse recurrent connections
                        + u_t_minus_1 * h_transfer @ (self.W_hh_mask * self.W_hh).T
                        + x[batch_idx, t] @ self.W_ih.T
                        + z_t_minus_1 @ self.W_zh.T
                        + self.noise) / self.tau
                h_t = h_t_minus_1 + dhdt * dt
                h[batch_idx, t, :] = h_t
                h_t_minus_1 = h_t
                u_t_minus_1 = u_t
                # compute h output here so that it can be passed to both
                # z (output unit) and itself on the next time step
                h_transfer = torch.tanh(h_t)
                z_t = h_transfer @ self.W_hz.T
                z[batch_idx, t, :] = z_t
                z_t_minus_1 = z_t
        return z, h
