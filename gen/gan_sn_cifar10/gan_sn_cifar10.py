import torch
import torch.nn as nn


from .components.modules.resblocks import GBlock
from .components import sngan_base


class GANSnCifar10(sngan_base.SNGANBaseGenerator):
    """Generator for SNGAN on cifar10."""
    def __init__(self):
        super().__init__(nz=128, ngf=256, bottom_width=4)

        self.l1 = nn.Linear(self.nz, (self.bottom_width**2) * self.ngf)
        self.block2 = GBlock(self.ngf, self.ngf, upsample=True)
        self.block3 = GBlock(self.ngf, self.ngf, upsample=True)
        self.block4 = GBlock(self.ngf, self.ngf, upsample=True)
        self.b5 = nn.BatchNorm2d(self.ngf)
        self.c5 = nn.Conv2d(self.ngf, 3, 3, 1, padding=1)
        self.activation = nn.ReLU(True)

        nn.init.xavier_uniform_(self.l1.weight.data, 1.0)
        nn.init.xavier_uniform_(self.c5.weight.data, 1.0)

    def forward(self, x):
        h = self.l1(x)
        h = h.view(x.shape[0], -1, self.bottom_width, self.bottom_width)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.b5(h)
        h = self.activation(h)
        h = torch.tanh(self.c5(h))
        return h
