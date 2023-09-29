import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import snntorch as snn
from snntorch import spikegen
from snntorch import utils
from snntorch import surrogate

# dataloader arguments
batch_size = 30
data_path = './data'

# Define a transform
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

trainset = datasets.CIFAR10(root=data_path, train=True,
                            download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size,
                         shuffle=True, num_workers=0)

testset = datasets.CIFAR10(root=data_path, train=False,
                           download=True, transform=transform)
testloader = DataLoader(testset, batch_size=batch_size,
                        shuffle=True, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'active device: {device}')

# Network Architecture
num_inputs = 32*32
num_outputs = 10

# Temporal Dynamics
num_steps = 25
beta = 0.99
spike_grad = surrogate.fast_sigmoid(slope=2)

net = nn.Sequential(nn.Conv2d(3, 200, 5),
                    nn.MaxPool2d(2, 2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    nn.Conv2d(200, 64, 5),
                    nn.MaxPool2d(2, 2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    nn.Flatten(),
                    nn.Linear(64*5*5, 100),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
                    nn.Linear(100, 10),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)
                    ).to(device)

# Load the network onto CUDA if available
net.to(device)

feats = {}
def hook_func(m, inp, outp):
    feats['activation'] = outp[0].detach()

def forward_pass(net, num_steps, data):
    mem_rec = []
    spk_rec = []
    hook_rec = []
    utils.reset(net)  # resets hidden states for all LIF neurons in net

    for step in range(num_steps):
        spk_out, mem_out = net(data)
        spk_rec.append(spk_out)
        mem_rec.append(mem_out)
        hook_rec.append(feats['activation'])

    return torch.stack(spk_rec), torch.stack(mem_rec), torch.stack(hook_rec)

def get_neuron_activity(net, data, spiking_layer_index=-1, neuron_id = 5):
    all_layers = [module for module in net.modules() if not isinstance(module, nn.Sequential) and module]
    all_leaky_layers = [module for module in net.modules() if type(module) == snn._neurons.leaky.Leaky]
    all_leaky_layers[spiking_layer_index].register_forward_hook(hook_func)

    spk_rec, mem_rec, hook_rec = forward_pass(net, num_steps, data)
    X = hook_rec.cpu().detach().numpy()#.mean(axis=0)

    print(X.shape)

    return X[:, neuron_id]


# load pretrained model
#mname = 'snn cifar10 70% acc.pt'
mname = 'trained_snn_bs=128_n_epochs=20_n_t_steps=128_reg_strength=2_1e_06.pt'
model_state = torch.load(os.path.join('pretrained', mname), map_location = device)

net.load_state_dict(model_state)
net.eval()

data, targets = next(iter(trainloader))
data = data.to(device)
activation_on_batch = get_neuron_activity(net, data, spiking_layer_index=-3, neuron_id = 5)
print(activation_on_batch)