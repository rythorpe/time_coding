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


def gaussian_func(x, center, width):
    return np.exp(-(x - center) ** 2 / (2 * width ** 2))
