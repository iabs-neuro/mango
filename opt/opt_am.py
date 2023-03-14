import torch
from time import perf_counter as tpc


def opt_am(model, x, lr=1.E-4, evals=30, eps=1.E-7):
    """Activation Maximization (AM) method."""
    t = tpc()

    x = x.detach().clone().to(model.device)
    x.requires_grad = True

    a_list = []

    for i in range(evals):
        model.ann(x[None])
        a = model.hook.a_mean
        G = (torch.autograd.grad(a, x))[0]
        G /= torch.sqrt(torch.mean(torch.mul(G, G))) + eps
        x = x + lr * G

        a_list.append(a.detach().to('cpu').numpy())

        if i == 0 or (i+1) % 10 == 0 or i == evals-1:
            text = f'am > '
            text += f'm {i+1:-7.1e} | '
            text += f't {tpc()-t:-9.3e} | '
            text += f'y {a_list[-1]:-11.4e}'
            text += ' <<< DONE' if i == evals-1 else ''
            print(text)

    return x.detach().clone().to(model.device)
