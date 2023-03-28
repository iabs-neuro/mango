import numpy as np
from protes import protes
import torch
from ttopt import TTOpt


def opt_protes(model, gen, evals, n=10, lim=3., with_score=False):
    """Activation Maximization with PROTES."""

    lim_a = -lim # Grid lower bound
    lim_b = +lim # Grid upper bound

    def ind_to_poi(z_index):
        z_index = np.array(z_index) # From jax to numpy
        z = z_index / (n - 1) * (lim_b - lim_a) + lim_a
        z = torch.tensor(z, dtype=torch.float32, device=model.device)
        return z

    def func(z_index):
        z = ind_to_poi(z_index)
        x = gen.run(z)
        s = gen.score(x).detach().to('cpu').numpy() if with_score else 0.
        a = model.run_target(x).detach().to('cpu').numpy()
        return a + 0.1 * s

    info = {}
    z_index = protes(func, gen.d, n, int(evals), is_max=True, log=True,
        info=info, with_info_i_opt_list=True)[0]
    return ind_to_poi(z_index), info
