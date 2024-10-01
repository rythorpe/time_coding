"""Optimization functions for ANN models."""

import numpy as np
import torch
from torch.optim import Optimizer


def diff_loss(output, target):
    return target - output


class RLS_opt(Optimizer):
    def __init__(self, params, n_params, alpha=0.5):
        # super().__init__()
        # NB: assumes that we are only optimizing output weights (W_hz)
        self.params = params
        self.P = torch.eye(n_params) / alpha

    def step(self, h_response):
        for h_dim in h_response.shape[0]:
            h_dim_response = h_response[[h_dim]]
            self.P = self.P - ((self.P @ h_dim_response.T @ h_dim_response @ self.P) /
                               (1 + h_dim_response @ self.P @ h_dim_response.T))
        for W in self.params:
            if W.requires_grad:
                W = W + W.grad @ self.P @ h_response
