"""ANN models."""

import numpy as np

import torch
from torch import nn


class RNN(nn.Module):
    def __init__(self, n_inputs=1, n_hidden=300, n_outputs=1,
                 p_rel_range=(0.1, 0.9)):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.tau = 0.01  # 10 ms
        self.tau_depr = 0.2  # 200 ms; taken from Mongillo et al. Science 2008
        self.tau_facil = 1.5  # 1.5 s
        self.beta = 18.0
        self._init_gain = 2.2 / np.mean(p_rel_range)
        # scale up gain due to decrease in baseline conn strength from p_rel
        self.gain = self._init_gain 
        self.activation_gain = 8.0
        self.activation_thresh = 0.5
        prob_c = 0.1

        # constant network parameters
        self.p_rel = torch.empty(n_hidden)
        # self.W_ih = torch.empty(n_hidden, n_inputs)
        self.W_ih = torch.ones(n_hidden, n_inputs)
        self.W_hh = torch.empty(n_hidden, n_hidden)
        # self.W_hh = nn.Parameter(torch.empty(n_hidden, n_hidden),
        #                          requires_grad=True)

        # varied network parameters
        self.presyn_scaling = nn.Parameter(torch.ones(n_hidden),
                                           requires_grad=False)
        self.W_hz = nn.Parameter(torch.empty(n_outputs, n_hidden),
                                 requires_grad=True)
        self.offset_hz = nn.Parameter(torch.zeros(n_outputs),
                                      requires_grad=True)

        # initialize release probabilities
        # bounds taken from Tsodyks & Markram PNAS 1997
        torch.nn.init.uniform_(self.p_rel, a=p_rel_range[0], b=p_rel_range[1])

        # initialize input weights
        # w_input_std = 1 / np.sqrt(n_hidden)
        # torch.nn.init.normal_(self.W_ih, mean=0.0, std=w_input_std)

        # initialize hidden weights
        w_hidden_std = 1 / np.sqrt(prob_c * n_hidden)
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
        self.register_buffer('noise', torch.zeros(self.n_hidden))

    def transfer_func(self, h, gain=8.0, thresh=0.5):
        '''Activation function for single-unit activity in hidden layer.

        Maximum slope occurs at thresh and takes a value of gain / 4.
        '''
        return torch.sigmoid(gain * (h - thresh))

    def forward(self, x, h_0, r_0=None, u_0=None, dt=0.001,
                return_deriv=False, include_stp=True,
                noise_tau=0.01, noise_std=0.0, include_corr_noise=False):

        # assuming batches x time x n_inputs
        batch_size, seq_len, _ = x.size()

        # create matrices for storing time-dependent state variables
        ext_in_all = torch.zeros(batch_size, seq_len, self.n_hidden)
        r_t_all = torch.zeros(batch_size, seq_len, self.n_hidden)
        u_t_all = torch.zeros(batch_size, seq_len, self.n_hidden)
        h_t_all = torch.zeros(batch_size, seq_len, self.n_hidden)
        z_t_all = torch.zeros(batch_size, seq_len, self.n_outputs)

        # NB: doesn't work on CUDA; due to FORCE training, h_0 is updated
        # regularly in time and therefore lives on the CPU
        for batch_idx in range(batch_size):
            n_t_minus_1 = torch.zeros(self.n_hidden)

            # each batch can theoretically have distinct initial conditions,
            # injected noise, or inputs
            if r_0 is None or include_stp is False:
                r_t_minus_1 = torch.ones(self.n_hidden)
            else:
                r_t_minus_1 = r_0[batch_idx, :]
            if u_0 is None or include_stp is False:
                # u_t_minus_1 = torch.ones(self.n_hidden)
                u_t_minus_1 = self.p_rel
            else:
                u_t_minus_1 = u_0[batch_idx, :]
            h_t_minus_1 = h_0[batch_idx, :]
            h_transfer = self.transfer_func(h_0[batch_idx, :],
                                            gain=self.activation_gain,
                                            thresh=self.activation_thresh)

            # begin integration over time
            for t_idx in range(0, seq_len):
                # noise; correct for non-stationarity in stochastic process
                # where variance scales proportional to the sampling rate 1/dt
                noise_scaling_fctr = np.sqrt(dt) / dt
                noise_sample = torch.randn(self.n_hidden)
                if include_corr_noise is True:
                    # add correlated noise, then apply corrected normalization
                    # factor
                    noise_sample = (
                        noise_sample +
                        torch.ones(self.n_hidden) * torch.randn(1)
                    ) / np.sqrt(2)

                dndt = (-n_t_minus_1 / noise_tau
                        + noise_std * noise_scaling_fctr * noise_sample)
                n_t = n_t_minus_1 + dndt * dt

                # total external input, including noise
                ext_in = x[batch_idx, t_idx, :] @ self.W_ih.T + n_t_minus_1
                ext_in_all[batch_idx, t_idx, :] = ext_in.clone()

                # pre-syn STP: depletion of resources (depression)
                if r_0 is None or include_stp is False:
                    # fix syn resources at init state to silence depression
                    drdt = torch.zeros_like(r_t_minus_1)
                else:
                    drdt = ((1 - r_t_minus_1) / self.tau_depr
                            - self.beta * u_t_minus_1 * r_t_minus_1 * h_transfer)
                r_t = r_t_minus_1 + drdt * dt
                r_t_all[batch_idx, t_idx, :] = r_t.clone()

                # pre-syn STP: augmentation of utilization (facilitation)
                if u_0 is None or include_stp is False:
                    # fix syn utilization at init state to silence facilitation
                    dudt = torch.zeros_like(u_t_minus_1)
                else:
                    dudt = ((self.p_rel - u_t_minus_1) / self.tau_facil
                            + self.beta * self.p_rel * (1 - u_t_minus_1) * h_transfer)
                u_t = u_t_minus_1 + dudt * dt
                u_t_all[batch_idx, t_idx, :] = u_t.clone()

                # calculate total transfer weight
                effective_weight = (r_t_minus_1 * u_t_minus_1 *
                                    self.presyn_scaling *
                                    self.gain * self.W_hh * self.W_hh_mask)

                # post-synaptic integration
                dhdt = (-h_t_minus_1
                        + h_transfer @ effective_weight.T
                        + ext_in) / self.tau
                h_t = h_t_minus_1 + dhdt * dt
                h_t_all[batch_idx, t_idx, :] = h_t.clone()

                # compute firing rate response (h) here so that it can be
                # passed to both z (output unit) on the current time step and
                # itself (recurrently) on the next time step
                h_transfer = self.transfer_func(h_t,
                                                gain=self.activation_gain,
                                                thresh=self.activation_thresh)
                z_t_all[batch_idx, t_idx, :] = (h_transfer @ self.W_hz.T
                                                + self.offset_hz)

                # save for next time step
                n_t_minus_1 = n_t
                r_t_minus_1 = r_t
                u_t_minus_1 = u_t
                h_t_minus_1 = h_t

        if return_deriv is True:
            return dhdt, drdt, dudt
        else:
            return ext_in_all, h_t_all, r_t_all, u_t_all, z_t_all
