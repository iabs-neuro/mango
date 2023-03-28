from collections import OrderedDict
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision


from .gen_gan_sn import GenGANSn
from .gen_gan_sn import GenGANSnDsc


# TODO: mode build_* functions and View class into separate modules


NAMES = ['gan-sn', 'gan-fc6', 'gan-fc7', 'gan-fc8', 'gan-fc6', 'gan-pool5']


class Gen:
    def __init__(self, name='gan-sn', device='cpu', sz=32):
        if not name in NAMES:
            raise ValueError(f'Gen name "{name}" is not supported')
        self.name = name

        self.device = device
        self.sz = sz

        root = os.path.dirname(__file__) + '/data'

        self.gen = None
        self.dsc = None

        if name == 'gan-sn':
            self.gen = GenGANSn()
            self.gen.load_state_dict(torch.load(f'{root}/gen_gan_sn.pkl',
                map_location='cpu'))

            self.dsc = GenGANSnDsc()
            self.dsc.load_state_dict(torch.load(f'{root}/gen_gan_sn_dsc.pkl',
                map_location='cpu'))

            self.d = 128

        if name == 'gan-fc6':
            self.gen, self.d = build_fc6()

        if name == 'gan-fc7':
            self.gen, self.d = build_fc7()

        if name == 'gan-fc8':
            self.gen, self.d = build_fc8()

        if name == 'gan-pool5':
            self.gen, self.d = build_pool5()

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

        if with_grad:
            x = self.gen(z)
        else:
            with torch.no_grad():
                x = self.gen(z)

        if self.name == 'gan-sn':
            return x if is_batch else x[0]

        # TODO: check that the mapping below is correct!

        # x = x[:, [2, 1, 0], :, :]
        m = torch.reshape(torch.tensor([123.0, 117.0, 104.0]), (1, 3, 1, 1))
        x = torch.clamp(x + m.to(x.device), 0, 255.0) / 255.0

        m = [0.485, 0.456, 0.406]
        v = [0.229, 0.224, 0.225]

        x = torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop(self.sz),
            torchvision.transforms.Normalize(m, v),
        ])(x)

        return x if is_batch else x[0]

    def run_back(self, x):
        # return z
        raise NotImplementedError('TOOD')

    def score(self, x, with_grad=False):
        if self.dsc is None:
            raise ValueError('Descriminator is not available')

        is_batch = len(x.shape) == 4
        if not is_batch:
            x = x[None]

        if with_grad:
            y = self.dsc(x).ravel()
        else:
            with torch.no_grad():
                y = self.dsc(x).ravel()

        return y if is_batch else y[0]


class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)


def build_fc6(url=None):
    if not url:
        url = 'https://onedrive.live.com/download?cid=9CFFF6BCB39F6829&resid=9CFFF6BCB39F6829%2145339&authkey=AC2rQMt7Obr0Ba4'

    gen = nn.Sequential(OrderedDict([
        ('defc7', nn.Linear(
            in_features=4096, out_features=4096, bias=True)),
        ('relu_defc7', nn.LeakyReLU(
            negative_slope=0.3, inplace=True)),
        ('defc6', nn.Linear(
            in_features=4096, out_features=4096, bias=True)),
        ('relu_defc6', nn.LeakyReLU(
            negative_slope=0.3, inplace=True)),
        ('defc5', nn.Linear(
            in_features=4096, out_features=4096, bias=True)),
        ('relu_defc5', nn.LeakyReLU(
            negative_slope=0.3, inplace=True)),
        ('reshape', View((-1, 256, 4, 4))),
        ('deconv5', nn.ConvTranspose2d(
            256, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ('relu_deconv5', nn.LeakyReLU(
            negative_slope=0.3, inplace=True)),
        ('conv5_1', nn.ConvTranspose2d(
            256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('relu_conv5_1', nn.LeakyReLU(
            negative_slope=0.3, inplace=True)),
        ('deconv4', nn.ConvTranspose2d(
            512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ('relu_deconv4', nn.LeakyReLU(
            negative_slope=0.3, inplace=True)),
        ('conv4_1', nn.ConvTranspose2d(
            256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('relu_conv4_1', nn.LeakyReLU(
            negative_slope=0.3, inplace=True)),
        ('deconv3', nn.ConvTranspose2d(
            256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ('relu_deconv3', nn.LeakyReLU(
            negative_slope=0.3, inplace=True)),
        ('conv3_1', nn.ConvTranspose2d(
            128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('relu_conv3_1', nn.LeakyReLU(
            negative_slope=0.3, inplace=True)),
        ('deconv2', nn.ConvTranspose2d(
            128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ('relu_deconv2', nn.LeakyReLU(
            negative_slope=0.3, inplace=True)),
        ('deconv1', nn.ConvTranspose2d(
            64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ('relu_deconv1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('deconv0', nn.ConvTranspose2d(
            32, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
    ]))

    gen.load_state_dict(load('fc6', url))
    return gen, gen[0].in_features


def build_fc7(url=None):
    if not url:
        url = 'https://onedrive.live.com/download?cid=9CFFF6BCB39F6829&resid=9CFFF6BCB39F6829%2145338&authkey=AJ0R-daUAVYjQIw'

    return build_fc6(url)


def build_fc8(url=None):
    if not url:
        url = 'https://onedrive.live.com/download?cid=9CFFF6BCB39F6829&resid=9CFFF6BCB39F6829%2145340&authkey=AKIfNk7s5MGrRkU'

    gen = nn.Sequential(OrderedDict([
        ('defc7', nn.Linear(
            in_features=1000, out_features=4096, bias=True)),
        ('relu_defc7', nn.LeakyReLU(
            negative_slope=0.3, inplace=True)),
        ('defc6', nn.Linear(
            in_features=4096, out_features=4096, bias=True)),
        ('relu_defc6', nn.LeakyReLU(
            negative_slope=0.3, inplace=True)),
        ('defc5', nn.Linear(
            in_features=4096, out_features=4096, bias=True)),
        ('relu_defc5', nn.LeakyReLU(
            negative_slope=0.3, inplace=True)),
        ('reshape', View(
            (-1, 256, 4, 4))),
        ('deconv5', nn.ConvTranspose2d(
            256, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ('relu_deconv5', nn.LeakyReLU(
            negative_slope=0.3, inplace=True)),
        ('conv5_1', nn.ConvTranspose2d(
            256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('relu_conv5_1', nn.LeakyReLU(
            negative_slope=0.3, inplace=True)),
        ('deconv4', nn.ConvTranspose2d(
            512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ('relu_deconv4', nn.LeakyReLU(
            negative_slope=0.3, inplace=True)),
        ('conv4_1', nn.ConvTranspose2d(
            256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('relu_conv4_1', nn.LeakyReLU(
            negative_slope=0.3, inplace=True)),
        ('deconv3', nn.ConvTranspose2d(
            256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ('relu_deconv3', nn.LeakyReLU(
            negative_slope=0.3, inplace=True)),
        ('conv3_1', nn.ConvTranspose2d(
            128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('relu_conv3_1', nn.LeakyReLU(
            negative_slope=0.3, inplace=True)),
        ('deconv2', nn.ConvTranspose2d(
            128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ('relu_deconv2', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('deconv1', nn.ConvTranspose2d(
            64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ('relu_deconv1', nn.LeakyReLU(negative_slope=0.3, inplace=True)),
        ('deconv0', nn.ConvTranspose2d(
            32, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
    ]))

    gen.load_state_dict(load('fc8', url))
    return gen, gen[0].in_features


def build_pool5(url=None):
    if not url:
        url = 'https://onedrive.live.com/download?cid=9CFFF6BCB39F6829&resid=9CFFF6BCB39F6829%2145337&authkey=AFaUAgeoIg0WtmA'

    gen = nn.Sequential(OrderedDict([
        ('Rconv6', nn.Conv2d(
            256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('Rrelu6', nn.LeakyReLU(
            negative_slope=0.3, inplace=True)),
        ('Rconv7', nn.Conv2d(
            512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('Rrelu7', nn.LeakyReLU(
            negative_slope=0.3, inplace=True)),
        ('Rconv8', nn.Conv2d(
            512, 512, kernel_size=(3, 3), stride=(1, 1))),
        ('Rrelu8', nn.LeakyReLU(
            negative_slope=0.3, inplace=True)),
        ('deconv5', nn.ConvTranspose2d(
            512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ('relu_deconv5', nn.LeakyReLU(
            negative_slope=0.3, inplace=True)),
        ('conv5_1', nn.ConvTranspose2d(
            256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('relu_conv5_1', nn.LeakyReLU(
            negative_slope=0.3, inplace=True)),
        ('deconv4', nn.ConvTranspose2d(
            512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ('relu_deconv4', nn.LeakyReLU(
            negative_slope=0.3, inplace=True)),
        ('conv4_1', nn.ConvTranspose2d(
            256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('relu_conv4_1', nn.LeakyReLU(
            negative_slope=0.3, inplace=True)),
        ('deconv3', nn.ConvTranspose2d(
            256, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ('relu_deconv3', nn.LeakyReLU(
            negative_slope=0.3, inplace=True)),
        ('conv3_1', nn.ConvTranspose2d(
            128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))),
        ('relu_conv3_1', nn.LeakyReLU(
            negative_slope=0.3, inplace=True)),
        ('deconv2', nn.ConvTranspose2d(
            128, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ('relu_deconv2', nn.LeakyReLU(
            negative_slope=0.3, inplace=True)),
        ('deconv1', nn.ConvTranspose2d(
            64, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
        ('relu_deconv1', nn.LeakyReLU(
            negative_slope=0.3, inplace=True)),
        ('deconv0', nn.ConvTranspose2d(
            32, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))),
    ]))

    gen.load_state_dict(load('pool5', url))
    return gen, gen[0].in_channels


def load(name, url):
    torchhome = torch.hub._get_torch_home()
    ckpthome = os.path.join(torchhome, 'checkpoints')
    os.makedirs(ckpthome, exist_ok=True)
    filepath = os.path.join(ckpthome, 'upconvGAN_%s.pt'%name)

    if not os.path.exists(filepath):
        torch.hub.download_url_to_file(url, filepath, hash_prefix=None,
            progress=False)

    return torch.load(filepath)
