import numpy as np
import os
import torch


from .gan_sn_cifar10 import GANSnCifar10
from .gan_sn_cifar10 import GANSnDscCifar10
from .vae_vq_cifar10 import VAEVqCifar10


NAMES = ['gan_sn', 'vae_vq']


class Gen:
    def __init__(self, name, data, device='cpu'):
        if not name in NAMES:
            raise ValueError(f'Gen name "{name}" is not supported')
        self.name = name

        self.data = data
        self.device = device

        self.load()

    def ind_to_poi(self, z_index):
        z = np.array(z_index) # From jax to numpy
        if not self.discrete:
            z = z / (self.n - 1) * (self.lim_b - self.lim_a) + self.lim_a
        z = torch.tensor(z, dtype=torch.float32, device=self.device)
        return z

    def load(self):
        fpath = os.path.dirname(__file__) + '/_data'
        os.makedirs(fpath, exist_ok=True)

        self.d = None          # Dimension
        self.n = None          # Grid size
        self.lim_a = None      # Grid lower limit
        self.lim_b = None      # Grid upper limit
        self.discrete = False  # If discrete latent space

        self.gen = None        # Generator
        self.dsc = None        # Decriminator
        self.enc = None        # Encoder

        if self.name == 'gan_sn':
            self.load_gan_sn(fpath)

        if self.name == 'vae_vq':
            self.load_vae_vq(fpath)

    def load_gan_sn(self, fpath):
        if self.data.name != 'cifar10':
            msg = 'Gen "gan_sn" is ready only for "cifar10"'
            raise NotImplementedError(msg)

        fpath = os.path.dirname(__file__) + '/gan_sn_cifar10/data'

        self.gen = GANSnCifar10()
        self.gen.load_state_dict(
            torch.load(f'{fpath}/gan_sn_cifar10.pkl',
            map_location='cpu'))
        self.gen.to(self.device)
        self.gen.eval()

        self.dsc = GANSnDscCifar10()
        self.dsc.load_state_dict(
            torch.load(f'{fpath}/gan_sn_dsc_cifar10.pkl',
            map_location='cpu'))
        self.dsc.to(self.device)
        self.dsc.eval()

        self.d = 128
        self.n = 64
        self.lim_a = -4.
        self.lim_b = +4.
        self.discrete = False

    def load_vae_vq(self, fpath):
        if self.data.name != 'cifar10':
            msg = 'Gen "vae_vq" is ready only for "cifar10"'
            raise NotImplementedError(msg)

        fpath = os.path.dirname(__file__) + '/vae_vq_cifar10/data'

        vae = VAEVqCifar10()
        vae.load_state_dict(
            torch.load(f'{fpath}/vae_vq_cifar10.pt',
            map_location='cpu'))
        vae.to(self.device)
        vae.eval()

        # TODO: move it into VAEVqCifar10 class

        def dec(z):
            m = z.shape[0]
            Z = torch.zeros(self.d * m, 1, dtype=torch.int64,
                device=self.device)
            for i in range(m):
                Z[i*self.d:(i+1)*self.d, 0] = z[i, :]
            v = vae._vq_vae.forward_spec(Z)
            return vae._decoder(v)

        def enc(x):
            vq_output_eval = vae._pre_vq_conv(vae._encoder(x))
            _, valid_quantize, _, _ = vae._vq_vae(vq_output_eval)
            z = vae._vq_vae.encoding_indices_.clone()[:, 0]
            z = [zz[None] for zz in torch.split(z, self.d)]
            z = torch.cat(z)
            return z

        self.gen = dec
        self.enc = enc

        self.d = vae.embedding_dim
        self.n = vae.num_embeddings
        self.lim_a = None
        self.lim_b = None
        self.discrete = True

    def run(self, z, with_grad=False):
        if self.gen is None:
            raise ValueError('Generator is not available')

        z, is_batch = self._inp(z, 2)

        if with_grad:
            x = self.gen(z)
        else:
            with torch.no_grad():
                x = self.gen(z)

        return x if is_batch else x[0]

    def rev(self, x, with_grad=False):
        if self.enc is None:
            raise ValueError('Encoder is not available')

        x, is_batch = self._inp(x, 4)

        if with_grad:
            z = self.enc(x)
        else:
            with torch.no_grad():
                z = self.enc(x)

        return z if is_batch else z[0]

    def score(self, x, with_grad=False):
        if self.dsc is None:
            raise ValueError('Descriminator is not available')

        x, is_batch = self._inp(x, 4)

        if with_grad:
            y = self.dsc(x).ravel()
        else:
            with torch.no_grad():
                y = self.dsc(x).ravel()

        return y if is_batch else y[0]

    def _inp(self, x, sh):
        is_batch = len(x.shape) == sh

        if not torch.is_tensor(x):
            x = torch.tensor(x)

        if not is_batch:
            x = x[None]

        x = x.to(self.device)

        return x, is_batch
