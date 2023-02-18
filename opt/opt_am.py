import torch


def opt_am(model, x, lr=1.E-4, evals=30, eps=1.E-7):
    """Метод анализа активации Activation Maximization (AM).

    Args:
        model (Model): нейронная сеть.
        x (torch.tensor): начальное приближение (вход для ИНС).
        lr (float): параметр скорости обучения.
        evals (int): количество итераций градиентного метода (бюджет).
        eps (float): параметр шума.

    Returns:
        Image: входное изображение для ИНС, максимизирующее активацию.

    """
    x = x.detach().clone().to(model.device)
    x.requires_grad = True

    model.model.eval()

    for i in range(evals):
        model.model(x)
        a = model.hook.a
        G = (torch.autograd.grad(a, x))[0]
        G /= torch.sqrt(torch.mean(torch.mul(G, G))) + eps
        x = x + lr * G

    x = x.detach().cpu()
    m = x.mean()
    s = x.std() or 1.E-8
    sat = 0.2
    br = 0.8
    x = x.sub(m).div(s).mul(sat).add(br).clamp(0., 1.)
    return x.squeeze(0)
