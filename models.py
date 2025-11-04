"""RNN model."""

import numpy as np

import torch


class RNN(torch.nn.Module):
    def __init__(self, n_hidden=300, n_outputs=1,
                 p_rel_range=(0.1, 0.9), conn_rule=None):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.tau = 0.01  # 10 ms
        self.tau_depr = 0.2  # 200 ms; taken from Mongillo et al. Science 2008
        self.tau_facil = 1.5  # 1.5 s
        self.beta = 50.0
        # self._init_gain = 1.0 / np.mean(p_rel_range)
        self._init_gain = 1.0
        # scale up gain due to decrease in baseline conn strength from p_rel
        self.gain = self._init_gain
        self.activation_gain = 8.0
        self.activation_thresh = 0.5
        prob_c = 0.10

        # varied network parameters
        # input -> hidden layer weights + offsets
        self.W_ih = torch.nn.Parameter(torch.empty(n_hidden, 1),
                                       requires_grad=True)
        self.offset_ih = torch.nn.Parameter(torch.zeros(n_hidden),
                                            requires_grad=True)
        # recurrent hidden layer postsynaptic weights
        self.W_hh = torch.nn.Parameter(torch.empty(n_hidden, n_hidden),
                                       requires_grad=True)
        # hidden -> output layer weights + offsets
        self.W_hz = torch.nn.Parameter(torch.empty(n_outputs, n_hidden),
                                       requires_grad=True)
        self.offset_hz = torch.nn.Parameter(torch.zeros(n_outputs),
                                            requires_grad=False)

        # initialize release probabilities; default bounds taken from
        # Tsodyks & Markram PNAS 1997
        # self.p_rel = torch.empty(n_hidden)
        # torch.nn.init.uniform_(self.p_rel, a=p_rel_range[0], b=p_rel_range[1])
        p_rel_mean = 0.35
        p_rel_std = 0.15
        p_rel = torch.randn(n_hidden) * p_rel_std + p_rel_mean
        resample = True
        while resample is True:
            stuff = torch.logical_or(p_rel <= 0, p_rel > 1)
            n_resample = stuff.sum()
            if n_resample > 0:
                p_rel[stuff] = torch.randn(n_resample) * p_rel_std + p_rel_mean
            else:
                resample = False
        self.p_rel = p_rel

        # scale all postsynaptic targets according to their presynaptic source
        # self.presyn_scaling = torch.ones(n_hidden)
        self.presyn_scaling = 1 / self.p_rel

        # initialize input weights
        torch.nn.init.normal_(self.W_ih, mean=0.0, std=1.0)

        # initialize hidden weights
        w_hidden_std = 1 / np.sqrt(prob_c * n_hidden)
        if conn_rule is None:
            torch.nn.init.normal_(self.W_hh, mean=0.0, std=w_hidden_std)
        else:
            with torch.no_grad():
                for target_idx in range(self.n_hidden):
                    source_weights = torch.randn(self.n_hidden) * w_hidden_std
                    # sort according to decending weight strength
                    weight_sort_idxs = torch.argsort(source_weights.abs(),
                                                     descending=True)
                    if conn_rule == 'p_rel_cluster':
                        # strong inter-connectivity between units of similar p_rel
                        order_metric = torch.abs(self.p_rel[target_idx] - self.p_rel)
                        # sources with p_rel similar to that of target will receive early position in sorted set
                        source_sort_idxs = order_metric.argsort(descending=False)
                    elif conn_rule == 'p_rel_anticluster':
                        # strong inter-connectivity between units of dissimilar p_rel
                        order_metric = torch.abs(self.p_rel[target_idx] - self.p_rel)
                        source_sort_idxs = order_metric.argsort(descending=True)
                    elif conn_rule == 'p_rel_corr':
                        # connection strength correlates with presynaptic p_rel
                        order_metric = self.p_rel.clone()
                        source_sort_idxs = order_metric.argsort(descending=True)
                    elif conn_rule == 'p_rel_anticorr':
                        # connection strength anticorrelates with presynaptic p_rel
                        order_metric = self.p_rel.clone()
                        source_sort_idxs = order_metric.argsort(descending=False)
                    # assign sorted source weights
                    self.W_hh[target_idx, source_sort_idxs] = source_weights[weight_sort_idxs]

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

        # create mask for specifying hidden subsets that map to distinct outputs
        # self.W_hz_mask = torch.ones_like(self.W_hz)
        self.W_hz_mask = torch.zeros_like(self.W_hz)
        n_sources_per_subset = self.n_hidden // n_outputs
        for output_idx in range(n_outputs):
            first_source_idx = n_sources_per_subset * output_idx
            last_source_idx = n_sources_per_subset * output_idx + n_sources_per_subset
            self.W_hz_mask[output_idx, first_source_idx:last_source_idx] = 1

        # create registered buffers (i.e., fancy attributes that need to live
        # on the same device as self
        # self.register_buffer('noise', torch.zeros(self.n_hidden))

    def transfer_func(self, h, gain=8.0, thresh=0.5):
        '''Activation function for single-unit activity in hidden layer.

        Maximum slope occurs at thresh and takes a value of gain / 4.
        '''
        return torch.sigmoid(gain * (h - thresh))

    def forward(self, I, h_0, r_0, u_0, n_0=None, dt=0.001,
                return_deriv=False, noise_tau=0.01, noise_std=0.0,
                include_corr_noise=False):

        # assuming input has shape (n_trials, n_times, n_hidden)
        batch_size, seq_len, _ = I.size()

        # create matrices for storing time-dependent state variables
        n_t_all = torch.zeros(batch_size, seq_len, self.n_hidden)
        r_t_all = torch.zeros(batch_size, seq_len, self.n_hidden)
        u_t_all = torch.zeros(batch_size, seq_len, self.n_hidden)
        h_t_all = torch.zeros(batch_size, seq_len, self.n_hidden)
        z_t_all = torch.zeros(batch_size, seq_len, self.n_outputs)

        # NB: doesn't work on CUDA; due to FORCE training, h_0 is updated
        # regularly in time and therefore lives on the CPU
        for trial_idx in range(batch_size):

            # each batch can theoretically have distinct initial conditions,
            # injected noise, or inputs
            if n_0 is None:
                n_t_minus_1 = noise_std * torch.randn(self.n_hidden)
            else:
                n_t_minus_1 = n_0[trial_idx, :]
            r_t_minus_1 = r_0[trial_idx, :]
            u_t_minus_1 = u_0[trial_idx, :]
            h_t_minus_1 = h_0[trial_idx, :]

            h_transfer = self.transfer_func(h_0[trial_idx, :],
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
                # correct for scaling of variance w.r.t. reference tau
                dndt = (-n_t_minus_1 / noise_tau
                        + noise_std * np.sqrt(1.0 / noise_tau)
                        * noise_scaling_fctr * noise_sample)
                n_t = n_t_minus_1 + dndt * dt
                n_t_all[trial_idx, t_idx, :] = n_t.clone()

                # pre-syn STP: depletion of resources (depression)
                drdt = ((1 - r_t_minus_1) / self.tau_depr
                        - self.beta * u_t_minus_1 * r_t_minus_1 * h_transfer)
                r_t = r_t_minus_1 + drdt * dt
                r_t_all[trial_idx, t_idx, :] = r_t.clone()

                # pre-syn STP: augmentation of utilization (facilitation)
                dudt = ((self.p_rel - u_t_minus_1) / self.tau_facil
                        + self.beta * self.p_rel * (1 - u_t_minus_1) * h_transfer)
                u_t = u_t_minus_1 + dudt * dt
                u_t_all[trial_idx, t_idx, :] = u_t.clone()

                # calculate total transfer weight
                effective_weight = (r_t_minus_1 * u_t_minus_1 *
                                    self.presyn_scaling *
                                    self.gain * self.W_hh * self.W_hh_mask)

                # post-synaptic integration
                ext_in = I[trial_idx, t_idx, :] @ self.W_ih.T + self.offset_ih + n_t_minus_1
                dhdt = (-h_t_minus_1
                        + h_transfer @ effective_weight.T
                        + ext_in) / self.tau
                h_t = h_t_minus_1 + dhdt * dt
                h_t_all[trial_idx, t_idx, :] = h_t.clone()

                # compute firing rate response (h) here so that it can be
                # passed to both z (output unit) on the current time step and
                # itself (recurrently) on the next time step
                h_transfer = self.transfer_func(h_t,
                                                gain=self.activation_gain,
                                                thresh=self.activation_thresh)
                output_weight = self.W_hz * self.W_hz_mask
                z_t_all[trial_idx, t_idx, :] = (h_transfer @ output_weight.T
                                                + self.offset_hz)

                # save for next time step
                n_t_minus_1 = n_t
                r_t_minus_1 = r_t
                u_t_minus_1 = u_t
                h_t_minus_1 = h_t

        if return_deriv is True:
            return dhdt, drdt, dudt
        else:
            return n_t_all, h_t_all, r_t_all, u_t_all, z_t_all
