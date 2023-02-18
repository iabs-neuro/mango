import numpy as np
from protes import protes
import torch
from ttopt import TTOpt


def opt_protes(model, gen, evals, n=10):
    """Activation Maximization with PROTES."""

    lim_a = -1. # Grid lower bound
    lim_b = +1. # Grid upper bound

    def func(z_index):
        z_index = np.array(z_index) # From jax to numpy
        z = z_index / (n - 1) * (lim_b - lim_a) + lim_a
        z = torch.tensor(z, dtype=torch.float32, device=model.device)
        x = gen.run(z)
        a = model.run_target(x)
        a = a.detach().to('cpu').numpy()
        return a

    z_index = protes(func, [n]*gen.d, evals, is_max=True, log=True)[0]
    z_index = np.array(z_index) # From jax to numpy
    z = z_index / (n - 1) * (lim_b - lim_a) + lim_a
    return torch.tensor(z, dtype=torch.float32, device=model.device)
