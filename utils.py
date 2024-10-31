"""Utility variables and functions."""

import numpy as np
import torch


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
