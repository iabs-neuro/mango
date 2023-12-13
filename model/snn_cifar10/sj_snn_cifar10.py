from spikingjelly.activation_based import surrogate, neuron, functional
from spikingjelly.activation_based.model import spiking_resnet
from spikingjelly.activation_based.model.spiking_resnet import SpikingResNet, BasicBlock

from collections import OrderedDict
import torch.nn as nn
import torch


def preprocess_train_sample_snn(T: int, x: torch.Tensor):
    # define how to process train sample before send it to model
    if len(x.shape) == 4:
        return x.unsqueeze(0).repeat(T, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]
    elif len(x.shape) == 3:
        return x.unsqueeze(0).repeat(T, 1, 1, 1)  # [C, H, W] -> [T, C, H, W]

class SJSNNCifar10(SpikingResNet):
    def __init__(self,
                 spiking_neuron=neuron.LIFNode,
                 surrogate_function=surrogate.ATan(),
                 detach_reset=True,
                 T=20):

        super(SJSNNCifar10, self).__init__(BasicBlock,
                                           [2, 2, 2, 2],
                                           num_classes=10,
                                           spiking_neuron=spiking_neuron,
                                           surrogate_function=surrogate_function,
                                           detach_reset=detach_reset)

        self.num_steps = T
        self.num_classes = 10

        self.named_layers_od = OrderedDict(self.named_modules())
        self.spiking_neuron = spiking_neuron

    def forward(self, x):
        xrep = preprocess_train_sample_snn(self.num_steps, x)
        return super(SJSNNCifar10, self).forward(xrep)


#snn = SJSNNCifar10()

