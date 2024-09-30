"""Optimization functions for ANN models."""

import numpy as np
import torch
from torch.optim import Optimizer


def diff_loss(output, target):
    return target - output


class RLS_opt(Optimizer):
    def __init__(self, params, alpha):
        super().__init__()
        n_params = len(params)
        self.W = params
        self.P = torch.eye(n_params) / alpha

    def step(self, h_out):
        self.P = self.P - ((self.P @ h_out @ h_out.T @ self.P) /
                           (1 + h_out.T @ self.P @ h_out.T))
        self.W = self.W + self.W.grad @ self.P @ h_out
