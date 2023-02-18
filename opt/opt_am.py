import torch


def opt_am(model, x, lr=1.E-4, evals=30, eps=1.E-7):
    """Activation Maximization (AM) method."""
    x = x.detach().clone().to(model.device)
    x.requires_grad = True

    for i in range(evals):
        model.ann(x)
        a = model.hook.a_mean
        G = (torch.autograd.grad(a, x))[0]
        G /= torch.sqrt(torch.mean(torch.mul(G, G))) + eps
        x = x + lr * G

    # Check these transformations:

    x = x.detach().cpu()
    m = x.mean()
    s = x.std() or 1.E-8
    sat = 0.2
    br = 0.8
    x = x.sub(m).div(s).mul(sat).add(br).clamp(0., 1.)

    return x
