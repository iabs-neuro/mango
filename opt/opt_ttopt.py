import torch
from ttopt import TTOpt


def opt_ttopt(model, gen, evals, rank=4):
    """Activation maximization with TTOpt."""

    def func(z):
        z = torch.tensor(z, dtype=torch.float32, device=model.device)
        x = gen.run(z)
        a = model.run_target(x)
        a = a.detach().to('cpu').numpy()
        return a

    tto = TTOpt(func, d=gen.d, a=-1., b=1., p=2, q=10, evals=evals,
        name='ttopt', with_cache=True, with_log=True)
    tto.maximize(rank)

    z = tto.x_min # Target latent vector
    return torch.tensor(z, dtype=torch.float32, device=model.device)
