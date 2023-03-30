import os
import torch


from .gan_sn import GANSn
from .gan_sn import GANSnDsc


NAMES = ['gan_sn']


class Gen:
    def __init__(self, name='gan_sn', device='cpu'):
        if not name in NAMES:
            raise ValueError(f'Gen name "{name}" is not supported')
        self.name = name

        self.device = device

        self.load()

    def load(self):
        self.gen = None  # Generator
        self.dsc = None  # Decriminator

        if self.name == 'gan_sn':
            fold = os.path.dirname(__file__) + '/gan_sn'

            self.gen = GANSn()
            self.gen.load_state_dict(torch.load(f'{fold}/data/gan_sn.pkl',
                map_location='cpu'))

            self.dsc = GANSnDsc()
            self.dsc.load_state_dict(torch.load(f'{fold}/data/gan_sn_dsc.pkl',
                map_location='cpu'))

            self.d = 128
            self.sz = 32

        if self.gen is not None:
            self.gen.to(self.device)
            self.gen.eval()

        if self.dsc is not None:
            self.dsc.to(self.device)
            self.dsc.eval()

    def run(self, z, with_grad=False):
        is_batch = len(z.shape) == 2
        if not is_batch:
            z = z[None]
        z = z.to(self.device)

        if with_grad:
            x = self.gen(z)
        else:
            with torch.no_grad():
                x = self.gen(z)

        return x if is_batch else x[0]

    def score(self, x, with_grad=False):
        if self.dsc is None:
            raise ValueError('Descriminator is not available')

        is_batch = len(x.shape) == 4
        if not is_batch:
            x = x[None]
        x = x.to(self.device)

        if with_grad:
            y = self.dsc(x).ravel()
        else:
            with torch.no_grad():
                y = self.dsc(x).ravel()

        return y if is_batch else y[0]
