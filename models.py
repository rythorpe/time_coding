"""ANN models."""

import numpy as np

import torch
from torch import nn


class RNN(nn.Module):
    def __init__(self, n_inputs=1, n_hidden=300, n_outputs=1,
                 echo_state=False):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.noise_std = 0.0  # 0.001
        self.tau = 0.01  # 10 ms
        self.tau_depr = 0.2  # 200 ms; taken from Mongillo et al. Science 2008
        self.tau_facil = 1.5  # 1.5 s
        # self.p_rel = 1.0
        self.beta = 10.0
        stp_gain_adjustment = 1 / (0.5 / (1 + self.beta * 0.5 * self.tau_depr))
        gain = 1.6 * stp_gain_adjustment
        print(f'adjusted gain: {gain}')
        prob_c = 0.10

        # constant network parameters
        self.p_rel = torch.empty(n_hidden)
        # self.tau_depr = torch.empty(n_hidden)
        self.W_ih = torch.empty(n_hidden, n_inputs)
        self.W_hh = torch.empty(n_hidden, n_hidden)

        # varied network parameters
        self.presyn_scaling = nn.Parameter(torch.ones(n_hidden),
                                           requires_grad=False)
        self.W_hz = nn.Parameter(torch.empty(n_outputs, n_hidden),
                                 requires_grad=True)
        # self.W_hz = torch.empty(n_outputs, n_hidden)

        # initialize release probabilities
        # Bounds taken from Tsodyks & Markram PNAS 1997
        torch.nn.init.uniform_(self.p_rel, a=0.1, b=0.90)
        # torch.nn.init.uniform_(self.tau_depr, a=0.05, b=1.0)

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
        self.W_hh_mask = torch.zeros(n_conns_possible)
        self.W_hh_mask[rand_conns] = 1
        self.W_hh_mask = torch.reshape(self.W_hh_mask, (n_hidden, n_hidden))

        # initialize output weights
        w_output_std = 1 / np.sqrt(n_hidden)
        torch.nn.init.normal_(self.W_hz, mean=0.0, std=w_output_std)

        # create registered buffers (i.e., fancy attributes that need to live
        # on the same device as self
        # self.register_buffer('h_0', torch.zeros(self.n_hidden))
        self.register_buffer('noise', torch.zeros(self.n_hidden))

    def forward(self, x, h_0, r_0=None, u_0=None, dt=0.001):
        # assuming batches x time x n_inputs
        batch_size, seq_len, _ = x.size()

        # create matrices for storing time-dependent state variables
        r_t_all = torch.zeros(batch_size, seq_len, self.n_hidden)
        u_t_all = torch.zeros(batch_size, seq_len, self.n_hidden)
        h_t_all = torch.zeros(batch_size, seq_len, self.n_hidden)
        z_t_all = torch.zeros(batch_size, seq_len, self.n_outputs)

        # NB: doesn't work on CUDA; due to FORCE training, h_0 is updated
        # regularly in time and therefore lives on the CPU
        for batch_idx in range(batch_size):
            # each batch can theoretically have distinct initial conditions,
            # injected noise, or inputs
            # r_t_minus_1 = torch.ones(self.n_hidden) * (0.5 / (1 + self.beta * 0.5 * self.tau_depr))
            if r_0 is None:
                r_t_minus_1 = torch.ones(self.n_hidden)
            else:
                r_t_minus_1 = r_0[batch_idx, :]
            if u_0 is None:
                r_t_minus_1 = torch.ones(self.n_hidden)
            else:
                u_t_minus_1 = u_0[batch_idx, :]
            h_t_minus_1 = h_0[batch_idx, :]
            h_transfer = torch.tanh(h_0[batch_idx, :])
            # begin integration over time
            for t_idx in range(0, seq_len):
                # init empty vector for current time step
                # dhdt = torch.zeros(self.n_hidden)
                self.noise.normal_(0, self.noise_std)

                # pre-syn STP: depletion of resources (depression)
                if r_0 is None:
                    # silence the effect of syn depression
                    r_t = torch.ones(self.n_hidden)
                else:
                    # drdt = ((1 - r_t_minus_1) / self.tau_depr
                    #         - self.beta * u_t_minus_1 * r_t_minus_1 *
                    #         (1 + h_transfer) / 2)
                    drdt = ((self.p_rel - r_t_minus_1) / self.tau_depr
                            - self.beta * r_t_minus_1 * (1 + h_transfer) / 2)
                    r_t = r_t_minus_1 + drdt * dt
                r_t_all[batch_idx, t_idx, :] = r_t.clone()

                # pre-syn STP: augmentation of utilization (facilitation)
                if u_0 is None:
                    # silence the effect of syn facilitation
                    u_t = torch.ones(self.n_hidden)
                else:
                    dudt = ((self.p_rel - u_t_minus_1) / self.tau_facil
                            + self.beta * (1 - u_t_minus_1) *
                            (1 + h_transfer) / 2)
                    u_t = u_t_minus_1 + dudt * dt
                u_t_all[batch_idx, t_idx, :] = u_t.clone()

                # calculate total transfer weight
                effective_weight = (r_t_minus_1 *
                                    self.presyn_scaling *
                                    self.W_hh * self.W_hh_mask)

                # post-synaptic integration
                dhdt = (-h_t_minus_1 +
                        x[batch_idx, t_idx, :] @ self.W_ih.T +
                        h_transfer @ effective_weight.T +
                        self.noise) / self.tau
                h_t = h_t_minus_1 + dhdt * dt
                h_t_all[batch_idx, t_idx, :] = h_t.clone()

                # compute firing rate response (h) here so that it can be
                # passed to both z (output unit) on the current time step and
                # itself (recurrently) on the next time step
                h_transfer = torch.tanh(h_t)
                z_t_all[batch_idx, t_idx, :] = h_transfer @ self.W_hz.T

                # save for next time step
                r_t_minus_1 = r_t
                u_t_minus_1 = u_t
                h_t_minus_1 = h_t

        return z_t_all, h_t_all, r_t_all, u_t_all
