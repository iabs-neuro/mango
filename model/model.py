import os
import requests
import torch
from urllib.parse import urlencode
import warnings


from .densenet import DenseNet


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

    def check(self, tst=True):
        data = self.data.dataloader_tst if tst else self.data.dataloader_trn
        n, m = 0, 0

        for x, l_real in data:
            x = x.to(self.device)
            y = self.run(x)
            l_pred = torch.argmax(y, axis=1).detach().to('cpu')
            m += (l_pred == l_real).sum()
            n += len(l_real)

        return n, m

    def load(self):
        self.net = None

        root = os.path.dirname(__file__)

        fpath = os.path.dirname(__file__) + '/_data'
        os.makedirs(fpath, exist_ok=True)

        if self.name == 'densenet':
            fpath += '/densenet.pt'

            if not os.path.isfile(fpath):
                url_data = 'https://disk.yandex.ru/d/ndE0NjV2G72skw'
                url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'
                url += urlencode(dict(public_key=url_data))
                url = requests.get(url).json()['href']
                with open(fpath, 'wb') as f:
                    f.write(requests.get(url).content)

            self.net = DenseNet()
            state_dict = torch.load(fpath, map_location='cpu')
            self.net.load_state_dict(state_dict)

        if self.name == 'vgg16':
            self.net = torch.hub.load('pytorch/vision:v0.10.0', self.name,
                weights=True)

        if self.net is not None:
            self.net.to(self.device)
            self.net.eval()

    def get_a(self):
        # Return activation of target neuron as a number
        a = self.hook.a_mean.detach().to('cpu').numpy()
        return float(a)

    def rmv_target(self, is_init=False):
        if is_init:
            self.hook_hand = []
        else:
            while len(self.hook_hand) > 0:
                self.hook_hand.pop().remove()
            self.hook_hand = []

        self.cl = None
        self.layer = None
        self.filter = None
        self.hook = None

    def run(self, x):
        is_batch = len(x.shape) == 4
        if not is_batch:
            x = x[None]

        with torch.no_grad():
            y = self.net(x)
            y = self.probs(y)

        return y if is_batch else y[0]

    def run_target(self, x):
        is_batch = len(x.shape) == 4
        if not is_batch:
            x = x[None]

        y = self.run(x)

        if self.cl is not None:
            res = y[:, self.cl]
        else:
            res = self.hook.a # TODO: check (self.hook.a_mean ?)

        return res if is_batch else res[0]

    def set_target(self, layer=None, filter=None, cl=None):
        if cl is not None and (layer is not None or filter is not None):
            raise ValueError('Please, set later+filter or class, not both')

        self.cl = cl

        if layer is None or filter is None:
            self.layer = None
            self.filter = None
            return

        self.layer = self.net.features[layer] # TODO: check
        if type(self.layer) != torch.nn.modules.conv.Conv2d:
            raise ValueError('We work only with conv layers')

        self.filter = filter
        if self.filter < 0 or self.filter >= self.layer.out_channels:
            raise ValueError('Filter does not exist')

        self.hook = AmHook(self.filter)
        self.hook_hand = [self.layer.register_forward_hook(self.hook.forward)]


class AmHook():
    def __init__(self, filter):
        self.filter = filter
        self.a = None
        self.a_mean = None

    def forward(self, module, inp, out):
        self.a = torch.mean(out[:, self.filter, :, :], dim=(1, 2))
        self.a_mean = torch.mean(out[:, self.filter, :, :])
