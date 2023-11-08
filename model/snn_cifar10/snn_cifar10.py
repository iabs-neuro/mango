from collections import OrderedDict

import torch
import torch.nn as nn
import snntorch as snn

from snntorch import surrogate
from snntorch import utils

class SNNCifar10(nn.Module):
    def __init__(self, beta=0.9, sigmoid_slope=10, num_steps=128, num_classes=10):
        super().__init__()

        spike_grad = surrogate.fast_sigmoid(slope=sigmoid_slope)

        self.features = nn.Sequential(
            OrderedDict(
                [
                    ('conv0', nn.Conv2d(3, 200, 5)),
                    ('pool0', nn.MaxPool2d(2, 2)),
                    ('spiking0', snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)),
                    ('conv1', nn.Conv2d(200, 64, 5)),
                    ('pool1', nn.MaxPool2d(2, 2)),
                    ('spiking1', snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)),
                    ('flatten', nn.Flatten()),
                    ('linear1', nn.Linear(64 * 5 * 5, 100)),
                    ('spiking2', snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)),
                    ('linear2', nn.Linear(100,10)),
                    ('relu0', nn.ReLU(inplace=True)),
                    ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                    ('final_spiking', snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True))
                ]
            )
        )

        self.num_steps = num_steps
        self.num_classes = num_classes

    def forward(self, data):
        num_steps = self.num_steps
        net = self.features
        mem_rec = []
        spk_rec = []
        utils.reset(net)  # resets hidden states for all LIF neurons in net

        for step in range(num_steps):
            spk_out, mem_out = net(data)
            spk_rec.append(spk_out)
            mem_rec.append(mem_out)

        return torch.stack(spk_rec), torch.stack(mem_rec)
