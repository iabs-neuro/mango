import torch


def opt_am(model, x, lr=0.1, iters=30, eps=1.E-7):
    """Метод анализа активации Activation Maximization (AM).

    Args:
        model (Model): нейронная сеть.
        x (torch.tensor): начальное приближение (вход для ИНС).
        lr (float): параметр скорости обучения.
        iters (int): количество итераций градиентного метода.
        eps (float): параметр шума.

    Returns:
        Image: входное изображение для ИНС, максимизирующее активацию.

    """
    x = x.detach().clone().to(model.device)
    x.requires_grad = True

    model.model.eval()

    class AmHook():
        def __init__(self, filter, shape, device):
            self.filter = filter
            self.A = None

        def forward(self, module, inp, out):
            self.A = torch.mean(out[:, self.filter, :, :])

    hook = AmHook(model.filter, x.shape, model.device)
    handlers = [model.layer.register_forward_hook(hook.forward)]
    for i in range(iters):
        model.model(x)
        A = hook.A
        G = (torch.autograd.grad(A, x))[0]
        G /= torch.sqrt(torch.mean(torch.mul(G, G))) + eps
        x = x + G * lr

    while len(handlers) > 0:
        handlers.pop().remove()

    v = x.detach().cpu()

    m = v.mean()
    s = v.std() or 1.E-8
    sat = 0.2
    br = 0.8
    v = v.sub(m).div(s).mul(sat).add(br).clamp(0., 1.)

    return v.squeeze(0)
