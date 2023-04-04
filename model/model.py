import numpy as np
import os
import sys
import torch
import warnings


from .densenet_cifar10 import DensenetCifar10


sys.path.append('..')
from utils import load_yandex


# To remove the warning of torchvision:
warnings.filterwarnings('ignore', category=UserWarning)


NAMES = ['densenet', 'vgg16']


class Model:
    def __init__(self, name, data, device='cpu'):
        if not name in NAMES:
            raise ValueError(f'Model name "{name}" is not supported')
        self.name = name

        self.data = data
        self.device = device

        self.probs = torch.nn.Softmax(dim=1)

        self.load()

        self.rmv_target(is_init=True)

    def check(self, tst=True, only_one_batch=False, with_target=False):
        data = self.data.dataloader_tst if tst else self.data.dataloader_trn
        n, m, a = 0, 0, []

        for x, l_real in data:
            x = x.to(self.device)
            y = self.run(x)
            l = torch.argmax(y, axis=1).detach().to('cpu')
            m += (l == l_real).sum()
            n += len(l)

            if with_target:
                a_cur = self.run_target(x)
                a.extend(list(a_cur.detach().to('cpu').numpy()))

            if only_one_batch:
                break

        return (n, m, np.array(a)) if with_target else (n, m)

    def load(self):
        fpath = os.path.dirname(__file__) + '/_data'
        os.makedirs(fpath, exist_ok=True)

        self.net = None

        if self.name == 'densenet':
            if self.data.name != 'cifar10':
                msg = 'Model "densenet" is ready only for "cifar10"'
                raise NotImplementedError(msg)

            fpath += '/densenet_cifar10.pt'

            if not os.path.isfile(fpath):
                load_yandex('https://disk.yandex.ru/d/ndE0NjV2G72skw', fpath)

            self.net = DensenetCifar10()
            state_dict = torch.load(fpath, map_location='cpu')
            self.net.load_state_dict(state_dict)

        if self.name == 'vgg16':
            if self.data.name != 'imagenet':
                msg = 'Model "vgg16" is ready only for "imagenet"'
                raise NotImplementedError(msg)

            # TODO: set path to data

            self.net = torch.hub.load('pytorch/vision:v0.10.0', self.name,
                weights=True)

        if self.net is not None:
            self.net.to(self.device)
            self.net.eval()

    def get_a(self):
        # Return activation of target neuron as a number
        a = self.hook.a_mean.detach().to('cpu').numpy()
        return float(a)

    def has_target(self):
        return self.c is not None or (self.l is not None and self.f is not None)

    def rmv_target(self, is_init=False):
        if is_init:
            self.hook_hand = []
        else:
            while len(self.hook_hand) > 0:
                self.hook_hand.pop().remove()
            self.hook_hand = []

        self.c = None
        self.l = None
        self.f = None
        self.hook = None

    def run(self, x):
        is_batch = len(x.shape) == 4
        if not is_batch:
            x = x[None]
        x = x.to(self.device)

        with torch.no_grad():
            y = self.net(x)
            y = self.probs(y)

        return y if is_batch else y[0]

    def run_pred(self, x):
        is_batch = len(x.shape) == 4
        if not is_batch:
            x = x[None]
        y = self.run(x).detach().to('cpu').numpy()

        c = np.argmax(y, axis=1)
        p = np.array([y[i, c_cur] for i, c_cur in enumerate(c)])
        l = [self.data.labels[c_cur] for c_cur in c]

        return (p, l) if is_batch else (p[0], l[0])

    def run_target(self, x):
        is_batch = len(x.shape) == 4
        if not is_batch:
            x = x[None]

        y = self.run(x)

        if self.c is not None:
            res = y[:, self.c]
        else:
            res = self.hook.a # TODO: check (self.hook.a_mean ?)

        return res if is_batch else res[0]

    def set_target(self, c=None, l=None, f=None):
        self.c = None
        self.l = None
        self.f = None

        if c is not None and (l is not None or f is not None):
            raise ValueError('Please, set class or later+filter, not both')

        if c is not None:
            self.c = int(c)
            return

        self.l = l
        self.f = f

        layer = self.net.features[l] # TODO: check
        if type(layer) != torch.nn.modules.conv.Conv2d:
            raise ValueError('We work only with conv layers')

        if self.f < 0 or self.f >= layer.out_channels:
            raise ValueError('Filter does not exist')

        self.hook = AmHook(self.f)
        self.hook_hand = [layer.register_forward_hook(self.hook.forward)]


class AmHook():
    def __init__(self, filter):
        self.filter = filter
        self.a = None
        self.a_mean = None

    def forward(self, module, inp, out):
        self.a = torch.mean(out[:, self.filter, :, :], dim=(1, 2))
        self.a_mean = torch.mean(out[:, self.filter, :, :])
