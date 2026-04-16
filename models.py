"""RNN model."""

import numpy as np
import torch

from utils import randn_cropped


class RNN(torch.nn.Module):
    def __init__(self, n_hidden=300, n_outputs=1,
                 p_rel_std=0.15):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.tau = 0.01  # 10 ms
        self.tau_depr = 0.2  # 200 ms; taken from Mongillo et al. Science 2008
        self.tau_facil = 1.5  # 1.5 s
        self.beta = 80.0
        self.gain = 1.0
        self.activation_gain = 8.0
        self.activation_thresh = 0.5
        prob_c = 0.1

        # varied network parameters
        # constant input to hidden layer
        self.offset_ih = torch.nn.Parameter(torch.zeros(n_hidden),
                                            requires_grad=True)
        # recurrent hidden layer postsynaptic weights
        self.W_hh = torch.nn.Parameter(torch.empty(n_hidden, n_hidden),
                                       requires_grad=True)
        # hidden -> output layer weights
        self.W_hz = torch.nn.Parameter(torch.empty(n_outputs, n_hidden),
                                       requires_grad=True)

        # initialize release probabilities; default bounds taken from
        # Tsodyks & Markram PNAS 1997
        # self.p_rel = torch.empty(n_hidden)
        # torch.nn.init.uniform_(self.p_rel, a=p_rel_range[0], b=p_rel_range[1])
        p_rel_mean = 0.35
        self.p_rel = randn_cropped(p_rel_mean, p_rel_std, (n_hidden,),
                                   lb=0.0, ub=1.0)

        # scale all postsynaptic targets according to their presynaptic source
        # correct for deminished presynaptic strength due to p_rel to place
        # each presynaptic unit on equal footing
        self.presyn_scaling = 1 / self.p_rel

        # initialize hidden weights
        # first, determine valence of presyn units for Dale's Law
        p_e = 0.8  # 4:1 E:I ratio
        e_units = torch.bernoulli(torch.ones(n_hidden) * p_e)
        presyn_valence = e_units * 2 - 1
        # tile and transpose s.t. valence is replicated
        # for each post-synaptic target
        presyn_valence = torch.tile(presyn_valence, (n_hidden, 1))

        # create mask for non-zero connections; this needs to be enforced
        # explicitly during training
        n_conns_possible = n_hidden ** 2
        n_conns_chosen = int(np.round(prob_c * n_hidden ** 2))
        rand_conns = np.random.choice(n_conns_possible, size=n_conns_chosen,
                                      replace=False)
        self.W_hh_mask = torch.zeros(n_conns_possible)
        self.W_hh_mask[rand_conns] = 1
        self.W_hh_mask = torch.reshape(self.W_hh_mask, (n_hidden, n_hidden))
        # incorporate valence into mask for element-wise multiplication
        self.W_hh_mask *= presyn_valence
        # sample magnitude of post-synaptic weight from cropped normal
        # distribution centered at zero
        w_hidden_std = 1 / np.sqrt(prob_c * n_hidden)
        weight_magnitude = torch.abs(torch.randn_like(self.W_hh) * w_hidden_std)
        # norm_mean = np.log10(0.15)
        # norm_std = 0.35
        # weight_magnitude = 10 ** (norm_mean + torch.randn_like(self.W_hh) * norm_std)
        # upscale mag of i_units for balance
        weight_magnitude[:, e_units == 0] *= p_e / (1 - p_e)
        # finally, set magnitude and valence of synaptic weight
        with torch.no_grad():
            self.W_hh.copy_(weight_magnitude * self.W_hh_mask)

        # initialize output weights
        # w_output_std = 1 / np.sqrt(n_hidden)
        # torch.nn.init.normal_(self.W_hz, mean=0.0, std=w_output_std)
        with torch.no_grad():
            self.W_hz[:] = 1 / (p_e * n_hidden / n_outputs)

        # create mask for specifying hidden subsets that map to distinct outputs
        self.W_hz_mask = torch.zeros_like(self.W_hz)
        n_sources_per_output = self.n_hidden // n_outputs
        for output_idx in range(n_outputs):
            first_source_idx = n_sources_per_output * output_idx
            last_source_idx = n_sources_per_output * output_idx + n_sources_per_output
            self.W_hz_mask[output_idx, first_source_idx:last_source_idx] = 1
        self.W_hz_mask[:, presyn_valence[0, :] == -1] = 0
        with torch.no_grad():
            self.W_hz.copy_(self.W_hz * self.W_hz_mask)

        # create registered buffers (i.e., fancy attributes that need to live
        # on the same device as self
        # self.register_buffer('noise', torch.zeros(self.n_hidden))

    def transfer_func(self, h, gain=8.0, thresh=0.5):
        '''Activation function for single-unit activity in hidden layer.

        Maximum slope occurs at thresh and takes a value of gain / 4.
        '''
        return torch.sigmoid(gain * (h - thresh))

    def forward(self, I, h_0, r_0, u_0, n_0=None, dt=0.001,
                return_deriv=False):

        # assuming input has shape (n_trials, n_times, n_hidden)
        batch_size, seq_len, _ = I.size()

        # create matrices for storing time-dependent state variables
        r_t_all = torch.zeros(batch_size, seq_len, self.n_hidden)
        u_t_all = torch.zeros(batch_size, seq_len, self.n_hidden)
        h_t_all = torch.zeros(batch_size, seq_len, self.n_hidden)
        z_t_all = torch.zeros(batch_size, seq_len, self.n_outputs)

        # NB: doesn't work on CUDA; due to FORCE training, h_0 is updated
        # regularly in time and therefore lives on the CPU
        for trial_idx in range(batch_size):

            # each batch can theoretically have distinct initial conditions,
            # injected noise, or inputs
            r_t_minus_1 = r_0[trial_idx, :]
            u_t_minus_1 = u_0[trial_idx, :]
            h_t_minus_1 = h_0[trial_idx, :]

            h_transfer = self.transfer_func(h_0[trial_idx, :],
                                            gain=self.activation_gain,
                                            thresh=self.activation_thresh)

            # begin integration over time
            for t_idx in range(0, seq_len):

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
                                    self.gain * self.W_hh)

                # post-synaptic integration
                ext_in = I[trial_idx, t_idx, :] + self.offset_ih
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

                z_t_all[trial_idx, t_idx, :] = h_transfer @ self.W_hz.T

                # save for next time step
                r_t_minus_1 = r_t
                u_t_minus_1 = u_t
                h_t_minus_1 = h_t

        if return_deriv is True:
            return dhdt, drdt, dudt
        else:
            return h_t_all, r_t_all, u_t_all, z_t_all


def OU_process(n_trials, n_times, n_dim, dt, noise_tau, noise_std,
               include_corr_noise=False):
    '''Simulate batch of independent trials of an OU process.'''

    if dt > 1e-2 * noise_tau:
        print('Warning: OU process may be unstable due to large dt. '
              'Ideally, dt << noise_tau.')

    n_t_all = torch.zeros(n_trials, n_times, n_dim)

    # set initial state by sampling from stationary distribution
    n_t_minus_1 = noise_std * torch.randn(n_trials, n_dim)

    for t_idx in range(n_times):
        noise_sample = torch.randn(n_trials, n_dim)
        if include_corr_noise is True:
            # add correlated noise sample, then correct for increase in variance
            noise_sample = (
                noise_sample +
                torch.ones(n_trials, n_dim) * torch.randn(n_trials, 1)
            ) / np.sqrt(2)

        # correct for scaling of variance w.r.t. reference tau
        # maintain stationarity about 2nd moment despite running summation;
        # noise scales with 1/sqrt(dt), not 1/dt
        sampling_rescaling = np.sqrt(dt) / dt
        dndt = (-n_t_minus_1 / noise_tau
                + noise_std * np.sqrt(2 / noise_tau) * noise_sample * sampling_rescaling)

        n_t = n_t_minus_1 + dndt * dt
        n_t_all[:, t_idx, :] = n_t.clone()

        # set prior state for next step in integration process
        n_t_minus_1 = n_t

    return n_t_all
