import numpy as np
import tqdm
import os
import pandas as pd
from os.path import join
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision

from .manager import MangoManager
import torch

from .snn.SpikingJelly.SJ_snn import SResNetTrainer
from .data.data_main import Data

METHODS = ['0_TT-exp', '1_TT-exp', '2_TT-exp']
ITERS = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 400, 500, 600, 700]

root = f'D:\\Projects\\mango_data\\SJ-SNN-T50'
gen = 'gan_sn'
dataname = 'cifar10'
model = 'sjsnn'
task = 'am'
kind = 'unit'
opt_args = {
    'opt_budget': 20000,
    'am_methods': METHODS
}
units = np.arange(64)

batch_size = 250
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_ = Data('cifar10', batch_trn=batch_size, batch_tst=batch_size, force_reload=False, root='D:\\Projects\\mango_data\\result')

print(data_.root)
data_path = os.path.join(data_.root, '_data', data_.name)

print(os.listdir(os.path.join(data_.root, '_data', data_.name)))

dataset = torchvision.datasets.CIFAR10(
    root=data_path,
    train=True,
    transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    )

dataset_test = torchvision.datasets.CIFAR10(
    root=data_path,
    train=False,
    transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    )

params = {
        'data-path': os.path.join(data_.root, '_data', data_.name),
        'batch-size': batch_size,
        'distributed': False,
        'cutmix-alpha': 1.0,
        'model': 'spiking_resnet18',
        'workers': 1,
        'T': 50,
        'train-crop-size': 32,
        'cupy': True,
        'epochs': 1000,
        'lr': 0.2,
        'random-erase': 0.1,
        'label-smoothing': 0.1,
        'momentum': 0.9
        #'resume': 'latest'
    }

trainer = SResNetTrainer()

parser = trainer.get_args_parser()
parser.add_argument('--distributed', type=bool, help="distributed")
args, _ = parser.parse_known_args()

#dataset, dataset_test, train_sampler, test_sampler = trainer.load_CIFAR10(args)

data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size
)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=batch_size
)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

train_losses = np.zeros(len(ITERS))
test_losses = np.zeros(len(ITERS))
train_acc = np.zeros(len(ITERS))
test_acc = np.zeros(len(ITERS))

train_losses_std = np.zeros(len(ITERS))
test_losses_std = np.zeros(len(ITERS))
train_acc_std = np.zeros(len(ITERS))
test_acc_std = np.zeros(len(ITERS))

for i, iter_ in tqdm.tqdm(enumerate(ITERS[:]), position=0):
    # ======================  upload model for this iter  ==========================
    model_path = f"D:\\Projects\\mango_data\\logs_t50\\checkpoint_{iter_}.pth"

    # we do not perform AM and use the network for inference only, manager params have no effect (except for model_path)
    manager = MangoManager(data=dataname,
                           gen=gen,
                           model=model,
                           task=task,
                           kind=kind,
                           cls=None,
                           unit=0,
                           layer='sn1',
                           opt_args=opt_args,
                           root=root,
                           device=device,
                           model_path=model_path)

    batch_train_loss = []
    batch_train_acc = []
    with torch.inference_mode():
        for j, (image, target) in enumerate(data_loader):
            image, target = image.to(device), target.to(device)
            output = manager.model.run(image)
            loss = criterion(output, target)
            acc1, acc5 = trainer.cal_acc1_acc5(output, target)

            batch_train_loss.append(loss.cpu().numpy())
            batch_train_acc.append(acc1.cpu().numpy())

    batch_test_loss = []
    batch_test_acc = []
    with torch.inference_mode():
        for j, (image, target) in enumerate(data_loader_test):
            image, target = image.to(device), target.to(device)
            output = manager.model.run(image)
            loss = criterion(output, target)
            acc1, acc5 = trainer.cal_acc1_acc5(output, target)

            batch_test_loss.append(loss.cpu().numpy())
            batch_test_acc.append(acc1.cpu().numpy())

    print('test:', np.mean(batch_test_loss),  np.mean(batch_test_acc))
    train_losses[i] = np.mean(batch_train_loss)
    train_acc[i] = np.mean(batch_train_acc)
    test_losses[i] = np.mean(batch_test_loss)
    test_acc[i] = np.mean(batch_test_acc)

    train_losses_std[i] = np.std(batch_train_loss)
    train_acc_std[i] = np.std(batch_train_acc)
    test_losses_std[i] = np.std(batch_test_loss)
    test_acc_std[i] = np.std(batch_test_acc)

np.savez(os.path.join(root, 'SJ-SNN T50 train acc'), train_acc)
np.savez(os.path.join(root, 'SJ-SNN T50 train loss'), train_losses)
np.savez(os.path.join(root, 'SJ-SNN T50 test acc'), test_acc)
np.savez(os.path.join(root, 'SJ-SNN T50 test loss'), test_losses)

np.savez(os.path.join(root, 'SJ-SNN T50 train acc std'), train_acc_std)
np.savez(os.path.join(root, 'SJ-SNN T50 train loss std'), train_losses_std)
np.savez(os.path.join(root, 'SJ-SNN T50 test acc std'), test_acc_std)
np.savez(os.path.join(root, 'SJ-SNN T50 test loss std'), test_losses_std)
