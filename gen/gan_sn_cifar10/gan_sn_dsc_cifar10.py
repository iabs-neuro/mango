import torch
import torch.nn as nn


from .components.modules.layers import SNLinear
from .components.modules.resblocks import DBlock
from .components.modules.resblocks import DBlockOptimized
from .components import sngan_base


class GANSnDscCifar10(sngan_base.SNGANBaseDiscriminator):
    """Discriminator for SNGAN on cifar10."""
    def __init__(self):
        super().__init__(ndf=128)

        self.block1 = DBlockOptimized(3, self.ndf)
        self.block2 = DBlock(self.ndf, self.ndf, downsample=True)
        self.block3 = DBlock(self.ndf, self.ndf, downsample=False)
        self.block4 = DBlock(self.ndf, self.ndf, downsample=False)
        self.l5 = SNLinear(self.ndf, 1)
        self.activation = nn.ReLU(True)

        nn.init.xavier_uniform_(self.l5.weight.data, 1.0)

    def forward(self, x):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.activation(h)
        h = torch.sum(h, dim=(2, 3))
        h = self.l5(h)
        return h
