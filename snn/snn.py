import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils

import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import itertools

# dataloader arguments
batch_size = 128
data_path = './data'

def print_and_log(inp):
    print(inp)
    with open(f'log_bs={batch_size}_num-steps={num_steps}_reg-strength={reg_strength}_n-epochs={num_epochs}'+'.txt', 'a') as f:
        f.write(inp+'\n')
        f.close()

# Define a transform
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))]
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
num_steps = 128
beta = 0.9
spike_grad = surrogate.fast_sigmoid(slope=5)

#Regularization
reg_strength = 2e-6

# Number of training epochs (iterations)
num_epochs = 40

net = nn.Sequential(nn.Conv2d(3, 200, 5),
                    nn.MaxPool2d(2, 2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, reset_mechanism="zero"),
                    nn.Conv2d(200, 64, 5),
                    nn.MaxPool2d(2, 2),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, reset_mechanism="zero"),
                    nn.Flatten(),
                    nn.Linear(64*5*5, 100),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, reset_mechanism="zero"),
                    nn.Linear(100, 10),
                    snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True, reset_mechanism="zero")
                    ).to(device)

# Load the network onto CUDA if available
net.to(device)


def forward_pass(net, num_steps, data):
    mem_rec = []
    spk_rec = []
    utils.reset(net)  # resets hidden states for all LIF neurons in net

    for step in range(num_steps):
        spk_out, mem_out = net(data)
        spk_rec.append(spk_out)
        mem_rec.append(mem_out)

    return torch.stack(spk_rec), torch.stack(mem_rec)


def batch_accuracy(loader, net, num_steps):
    with torch.no_grad():
        total = 0
        acc = 0
        net.eval()

    loader = iter(loader)
    for data, targets in loader:
        data = data.to(device)
        targets = targets.to(device)
        spk_rec, _ = forward_pass(net, num_steps, data)

        acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
        total += spk_rec.size(1)

    return acc / total


data, targets = next(iter(trainloader))
data = data.to(device)
targets = targets.to(device)
spk_rec, mem_rec = forward_pass(net, num_steps, data)

loss_fn = SF.ce_rate_loss()
loss_val = loss_fn(spk_rec, targets)
regularizer = SF.reg.l1_rate_sparsity(Lambda=reg_strength)

print_and_log(f"The loss from an untrained network is {loss_val.item():.3f}")

acc = SF.accuracy_rate(spk_rec, targets)
test_acc = batch_accuracy(testloader, net, num_steps)

print_and_log(f"The total accuracy on the test set is: {test_acc * 100:.2f}%")
print_and_log(f"The accuracy of a single batch using an untrained network is {acc*100:.3f}%")


optimizer = torch.optim.Adam(net.parameters(), lr=0.5e-3, betas=(0.9, 0.999))
loss_hist = []
test_acc_hist = []
counter = 0
running_loss = 0.0
checkpoint = 200

# Outer training loop
for epoch in range(num_epochs):

    print_and_log('*********************************')
    print_and_log(f'             epoch {epoch + 1}         ')
    print_and_log('*********************************')
    # Training loop
    for data, targets in iter(trainloader):
        data = data.to(device)
        targets = targets.to(device)

        # forward pass
        net.train()
        spk_rec, _ = forward_pass(net, num_steps, data)

        # initialize the loss & sum over time
        loss_val = loss_fn(spk_rec, targets) + regularizer(spk_rec)

        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        # Store loss history for future plotting
        loss_hist.append(loss_val.item())

        # Test set
        running_loss += loss_val.item()

        if counter % checkpoint == checkpoint - 1:
            with torch.no_grad():
                net.eval()

                # Test set forward pass
                test_acc = batch_accuracy(testloader, net, num_steps)
                # train_acc = batch_accuracy(trainloader, net, num_steps)
                print_and_log(f'Iteration {counter + 1}:')
                # print(f'Train Acc: {train_acc * 100:.2f}%')
                print_and_log(f'Train loss: {running_loss / checkpoint:.3f}')
                print_and_log(f'Test Acc: {test_acc * 100:.2f}%\n')

                # test_acc_hist.append(test_acc.item())
                running_loss = 0.0

        counter += 1
        
torch.save(net.state_dict(), f'trained-snn_bs={batch_size}_n-epochs={num_epochs}_n-t-steps={num_steps}_reg-strength={reg_strength}.pt')