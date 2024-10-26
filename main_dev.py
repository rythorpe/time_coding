"""Main development script for project."""

# %%
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn

from utils import get_device, gaussian
from models import RNN
from train import test, train, set_optimimal_w_out


# set meta-parameters
# device = get_device()
device = 'cpu'
# for reproducibility while troubleshooting; numpy is for model sparse conns
torch.random.manual_seed(95214)
np.random.seed(35107)

# define parameter sweep
n_samp = 2
params = {'n_outputs': np.linspace(5, 25, n_samp),
          'targ_std': np.linspace(0.005, 0.025, n_samp)}
xx, yy = np.meshgrid(params['n_outputs'], params['targ_std'])
param_vals = [pt for pt in zip(xx.flatten(), yy.flatten())]

for sample_idx, param_sample in enumerate(param_vals):
    n_outputs, targ_std = int(param_sample[0]), param_sample[1]

    # instantiate model, loss function, and optimizer
    # n_inputs, n_hidden, n_outputs = 1, 300, 10
    n_inputs, n_hidden = 1, 300
    model = RNN(n_inputs=n_inputs, n_hidden=n_hidden,
                n_outputs=n_outputs, echo_state=False)
    model.to(device)
    print(model)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # set parameters
    # simulation parameters
    dt = 1e-3  # 1 ms
    tstop = 1.1  # 1 sec
    times = np.arange(-0.1, tstop, dt)
    n_times = len(times)

    # define inputs (for contextual modulation / recurrent perturbations)
    n_batches = 1
    inputs = torch.zeros((n_batches, n_times, n_inputs))

    # define output targets
    output_delays = np.linspace(0.1, tstop - 0.1, n_outputs)  # w/ margins
    targets = torch.zeros((n_batches, n_times, n_outputs))
    for output_idx, center in enumerate(output_delays):
        targets[0, :, output_idx] = torch.tensor(gaussian(times, center, targ_std))


    # set initial conditions of recurrent units fixed across iterations of
    # training and testing
    h_0 = (torch.rand(n_hidden) * 2) - 1  # uniform in (-1, 1)
    h_0 = torch.tile(h_0, (n_batches, 1))  # replicate for each batch

    # run opt routine
    # move to desired device
    inputs, targets, h_0 = inputs.to(device), targets.to(device), h_0.to(device)

    # plot model output before training
    test(inputs, targets, times, model, loss_fn, h_0=h_0)

    # train model weights
    max_iter = 300
    convergence_reached = False
    loss_per_iter = list()
    for t in range(max_iter):
        print(f"Iteration {t + 1}")
        loss, param_dist = train(inputs, targets, times, model, loss_fn, optimizer, h_0=h_0)
        loss_per_iter.append(loss)
        if param_dist < 3e-4:
            convergence_reached = True
            break
    print(f"Trial {sample_idx} training complete!!")
    if not convergence_reached:
        print(f"Warning: didn't converge (param_dist={param_dist})!!")

    plt.figure()
    plt.plot(loss_per_iter)
    plt.xlabel('iteration')
    plt.ylabel('loss')

    # investigate fitted model
    # plot model output after training
    test(inputs, targets, times, model, loss_fn, h_0=h_0)
    # solve for optimal model output weights given hidden unit responses
    set_optimimal_w_out(inputs, targets, times, model, loss_fn, h_0=h_0)
