import sys

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils

from ..model.snn_cifar10.snn_cifar10 import SNNCifar10

dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'active device: {device}')

# dataloader arguments
batch_size = 64
data_path = './data'

# Define a transform
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))]
)

trainset = datasets.CIFAR10(root=data_path,
                            train=True,
                            download=True,
                            transform=transform)

#trainset.train_data.to(device)

trainloader = DataLoader(trainset,
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=0,
                         pin_memory=True)

testset = datasets.CIFAR10(root=data_path,
                           train=False,
                           download=True,
                           transform=transform)

testloader = DataLoader(testset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=0,
                        pin_memory=True)


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Network Architecture
num_inputs = 32*32
num_outputs = 10

# Temporal Dynamics
num_steps = 100
beta = 0.9
sigmoid_slope = 10
spike_grad = surrogate.fast_sigmoid(slope=sigmoid_slope)

# Regularization
reg_strength = 0.0
correct_rate = 0.33

# Number of training epochs (iterations)
num_epochs = 20

hyperparams = {
    'num_steps': num_steps,
    'beta': beta,
    'l1_rate_sparsity': reg_strength,
    'correct_rate': correct_rate,
    'num_epochs': num_epochs
}

def print_and_log(inp, hyperparams=hyperparams):
    print(inp)
    logname = "".join([f'{hname}={hparam}_' for (hname, hparam) in hyperparams.items()])
    logname = 'log_' + logname[:-1] + '.txt'
    with open(logname, 'a') as f:
        f.write(inp+'\n')
        f.close()


msnn = SNNCifar10(beta=beta, sigmoid_slope=sigmoid_slope, num_steps=num_steps, num_classes=len(classes))
net = msnn.features
# Load the network onto CUDA if available
net.to(device)


def batch_accuracy(loader, model):
    with torch.no_grad():
        total = 0
        acc = 0
        model.features.eval()

    loader = iter(loader)
    for data, targets in loader:
        data = data.to(device)
        targets = targets.to(device)
        spk_rec = model.forward(data)

        acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
        total += spk_rec.size(1)

    return acc / total

'''
data, targets = next(iter(trainloader))
data = data.to(device)
targets = targets.to(device)
spk_rec, mem_rec = msnn.forward(data, return_membrane=True)

loss_val = loss_fn(spk_rec, targets)

print_and_log(f"The loss from an untrained network is {loss_val.item():.3f}")

acc = SF.accuracy_rate(spk_rec, targets)
test_acc = batch_accuracy(testloader, msnn)

print_and_log(f"The total accuracy on the test set is: {test_acc * 100:.2f}%")
print_and_log(f"The accuracy of a single batch using an untrained network is {acc*100:.3f}%")
'''
loss_fn = SF.mse_count_loss(correct_rate=correct_rate, incorrect_rate=0)
regularizer = SF.reg.l1_rate_sparsity(Lambda=reg_strength)
optimizer = torch.optim.Adam(net.parameters(), lr=0.5e-3, betas=(0.9, 0.999))
loss_hist = []
test_loss_hist = []
counter = 0
running_loss = 0.0
test_running_loss = 0.0
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
        spk_rec = msnn.forward(data)

        # initialize the loss
        loss_val = loss_fn(spk_rec, targets) + regularizer(spk_rec)

        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        # Store loss history for future plotting
        loss_hist.append(loss_val.item())
        running_loss += loss_val.item()

        if counter % checkpoint == checkpoint - 1:
            with torch.no_grad():
                net.eval()
                # Test set forward pass
                for data, targets in iter(testloader):
                    data = data.to(device)
                    targets = targets.to(device)
                    spk_rec = msnn.forward(data)

                    # initialize the loss & sum over time
                    test_loss_val = loss_fn(spk_rec, targets) + regularizer(spk_rec)
                    test_running_loss += test_loss_val.item()

                    # Store loss history for future plotting
                    test_loss_hist.append(test_loss_val.item())

                # test_acc = batch_accuracy(testloader, msnn)
                # train_acc = batch_accuracy(trainloader, msnn)
                print_and_log(f'Iteration {counter + 1}:')
                # print(f'Train Acc: {train_acc * 100:.2f}%')
                print_and_log(f'Train loss: {running_loss / checkpoint:.3f}')
                print_and_log(f'Test loss: {test_running_loss / checkpoint:.3f}')
                #print_and_log(f'Test Acc: {test_acc * 100:.2f}%\n')

                # test_acc_hist.append(test_acc.item())
                running_loss = 0.0
                test_running_loss = 0.0

        counter += 1

test_acc = batch_accuracy(testloader, msnn)
print_and_log(f'Test Acc: {test_acc * 100:.2f}%\n')

netname = "trained-snn_".join([f'{hname}={hparam}_' for (hname, hparam) in hyperparams.items()])
torch.save(net.state_dict(), netname[:-1] + '.pt')
