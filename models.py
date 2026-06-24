"""RNN model."""

import numpy as np
import torch

from utils import randn_cropped


class RNN(torch.nn.Module):
    def __init__(self, n_hidden=300, n_outputs=1,
                 p_rel_params=(0.35, 0.15)):
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
        p_rel_mean, p_rel_std = p_rel_params
        self.p_rel = randn_cropped(p_rel_mean, p_rel_std, (n_hidden,),
                                   lb=0.0, ub=1.0)

        # scale all postsynaptic targets according to their presynaptic source
        # correct for deminished presynaptic strength due to p_rel to place
        # each presynaptic unit on equal footing
        self.presyn_scaling = 1 / self.p_rel

        # set tau_syn values for alt filtering model
        u_rand = torch.rand(n_hidden)
        f_x_rand = torch.rand(n_hidden)
        self.tau_syn_depr = (self.tau_depr / (1 + (self.tau_depr *
                                                   self.beta *
                                                   u_rand *
                                                   f_x_rand)))
        self.tau_syn_facil = (self.tau_facil / (1 + (self.tau_facil *
                                                     self.beta *
                                                     self.p_rel *
                                                     f_x_rand)))

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
        # upscale mag of i_units for balance
        weight_magnitude[:, e_units == 0] *= p_e / (1 - p_e)
        # finally, set magnitude and valence of synaptic weight
        with torch.no_grad():
            self.W_hh.copy_(weight_magnitude * self.W_hh_mask)

        # initialize output weights
        n_sources_per_output = self.n_hidden // n_outputs
        w_output_std = 1 / np.sqrt(n_sources_per_output)
        torch.nn.init.normal_(self.W_hz, mean=0.0, std=w_output_std)

        # create mask for specifying hidden subsets that map to distinct outputs
        self.W_hz_mask = torch.zeros_like(self.W_hz)
        for output_idx in range(n_outputs):
            first_source_idx = n_sources_per_output * output_idx
            last_source_idx = n_sources_per_output * output_idx + n_sources_per_output
            self.W_hz_mask[output_idx, first_source_idx:last_source_idx] = 1

        # create registered buffers (i.e., fancy attributes that need to live
        # on the same device as self
        # self.register_buffer('noise', torch.zeros(self.n_hidden))

    def transfer_func(self, h, gain=8.0, thresh=0.5):
        '''Activation function for single-unit activity in hidden layer.

        Maximum slope occurs at thresh and takes a value of gain / 4.
        '''
        return torch.sigmoid(gain * (h - thresh))

    def forward(self, I, h_0, r_0, u_0, dt=0.001, model_version='stp'):

        # assuming input has shape (n_trials, n_times, n_hidden)
        batch_size, seq_len, _ = I.size()

        # create matrices for storing time-dependent state variables
        r_ = torch.zeros(batch_size, seq_len, self.n_hidden)
        u_ = torch.zeros(batch_size, seq_len, self.n_hidden)
        h_ = torch.zeros(batch_size, seq_len, self.n_hidden)
        z_ = torch.zeros(batch_size, seq_len, self.n_outputs)

        # each batch can theoretically have distinct initial conditions,
        # injected noise, or inputs
        r_t_minus_1 = r_0
        u_t_minus_1 = u_0
        h_t_minus_1 = h_0

        f_t_minus_1 = self.transfer_func(h_0,
                                         gain=self.activation_gain,
                                         thresh=self.activation_thresh)

        # begin integration over time
        for t_idx in range(0, seq_len):

            if model_version == 'stp':
                # pre-syn STP: depletion of resources (depression)
                drdt = ((1 - r_t_minus_1) / self.tau_depr
                        - self.beta * u_t_minus_1 * r_t_minus_1 * f_t_minus_1)
                r_t = r_t_minus_1 + drdt * dt

                # pre-syn STP: augmentation of utilization (facilitation)
                dudt = ((self.p_rel - u_t_minus_1) / self.tau_facil
                        + self.beta * self.p_rel * (1 - u_t_minus_1) * f_t_minus_1)
                u_t = u_t_minus_1 + dudt * dt

                # calculate total reccurent input (post-syn current)
                reccurent_input = self.gain * (self.presyn_scaling *
                                               r_t_minus_1 *
                                               u_t_minus_1 *
                                               f_t_minus_1) @ self.W_hh.T
            elif model_version == 'alt_stp':
                # fast timescale filter
                drdt = (-r_t_minus_1 + f_t_minus_1) / self.tau_syn_depr
                r_t = r_t_minus_1 + drdt * dt

                # slow timescale filter
                dudt = (-u_t_minus_1 + r_t_minus_1) / self.tau_syn_facil
                u_t = u_t_minus_1 + dudt * dt

                # calculate total reccurent input (post-syn current)
                reccurent_input = self.gain * u_t_minus_1 @ self.W_hh.T

            # post-synaptic integration
            ext_in = I[:, t_idx, :] + self.offset_ih
            dhdt = (-h_t_minus_1
                    + reccurent_input
                    + ext_in) / self.tau
            h_t = h_t_minus_1 + dhdt * dt

            # compute firing rate response of hidden units here so that it can be
            # passed to both z (output units) on the current time step and
            # itself (recurrently) on the next time step
            f_t = self.transfer_func(h_t,
                                     gain=self.activation_gain,
                                     thresh=self.activation_thresh)
            output_weight = self.W_hz * self.W_hz_mask
            z_[:, t_idx, :] = f_t @ output_weight.T

            # update state in time
            r_t_minus_1 = r_t
            u_t_minus_1 = u_t
            h_t_minus_1 = h_t
            f_t_minus_1 = f_t

            # store timeseries
            r_[:, t_idx, :] = r_t.detach().clone()
            u_[:, t_idx, :] = u_t.detach().clone()
            h_[:, t_idx, :] = h_t.detach().clone()

        return h_, r_, u_, z_


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
