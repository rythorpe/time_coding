"""Utility variables and functions."""

import numpy as np
import scipy
import torch

import matplotlib.pyplot as plt


def get_device():
    # Get cpu, gpu or mps device for training.
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    return device


def gaussian(x, center, width):
    return np.exp(-(x - center) ** 2 / (2 * width ** 2))


def get_gaussian_targets(n_batches, n_outputs, times, targ_std):
    tstop = times[-1]
    n_times = len(times)
    # tile center of target delays spanning sim duration (minus margins)
    delta_delay = (tstop - 0.1) / n_outputs
    output_delays = np.arange(delta_delay, tstop - 0.1 + delta_delay,
                              delta_delay)
    targets = torch.zeros((n_batches, n_times, n_outputs))
    for output_idx, center in enumerate(output_delays):
        targets[0, :, output_idx] = torch.tensor(gaussian(times, center,
                                                          targ_std))
    return targets


def get_random_targets(model_class, inputs, model_dims, times, n_opt_basis=10,
                       device='cpu', plot=False):
    n_batches = inputs.shape[0]
    n_inputs, n_hidden, n_outputs = model_dims
    dt = times[1] - times[0]
    model = model_class(n_inputs=n_inputs, n_hidden=n_hidden,
                        n_outputs=n_outputs, echo_state=False)
    
    h_0 = (torch.rand(n_hidden) * 2) - 1  # uniform in (-1, 1)
    h_0 = torch.tile(h_0, (n_batches, 1))  # replicate for each batch

    model.to(device)
    model.eval()

    with torch.no_grad():
        # Compute prediction error
        outputs, h_t = model(inputs, h_0=h_0, dt=dt)
        transfer_func = torch.nn.Tanh()
        h_transfer = transfer_func(h_t)

    # 1st (and only) batch, 1st third of recurrent trajectories
    h_transfer_subset = np.array(h_transfer[0, :, :n_hidden // 3])
    cov = np.cov(h_transfer_subset, rowvar=False)
    eigvals, eigvecs = scipy.linalg.eig(cov)
    sort_idxs = np.argsort(eigvals)[-1::-1] # reverse for descending order
    eigvals, eigvecs = eigvals[sort_idxs], eigvecs[:, sort_idxs]
    opt_basis = h_transfer_subset @ eigvecs[:, :n_opt_basis]  # time x basis func

    if plot:
        fig, axes = plt.subplots(1, 2)
        axes[0].scatter(np.arange(n_opt_basis) + 1, eigvals[:n_opt_basis], c='r')
        axes[0].scatter(np.arange(n_opt_basis, len(eigvals)) + 1, eigvals[n_opt_basis:], c='b')
        axes[0].set_ylabel('eigenvalue')
        axes[1].plot(times, opt_basis)
        axes[1].set_ylabel('PC embedding')
        fig.show()

    return h_transfer_subset, opt_basis
