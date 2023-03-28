from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
import torchvision
import urllib
import yaml


NAMES = ['mnist', 'mnistf', 'cifar10', 'imagenet']


class Data:
    def __init__(self, name, batch_trn=256, batch_tst=32):
        if not name in NAMES:
            raise ValueError(f'Dataset name "{name}" is not supported')
        self.name = name
        self.batch_trn = batch_trn
        self.batch_tst = batch_tst

        self.load_shape()
        self.load_labels()
        self.load_transform()
        self.load_data()

    def get(self, i=None, tst=False):
        data = self.data_tst if tst else self.data_trn

        if i is None:
            i = torch.randint(len(data), size=(1,)).item()
        else:
            i = min(i, len(data)-1)

        x, c = data[i]
        return x, c

    def img_load(self, fpath, device='cpu'):
        img = Image.open(fpath)
        return self.transform(img).to(device)

    def img_rand(self, device='cpu'):
        pix = np.random.rand(self.sz, self.sz, self.ch) * 255
        img = Image.fromarray(pix.astype('uint8')).convert('RGB')
        return self.transform(img).to(device)

    def load_data(self):
        fpath = os.path.dirname(__file__) + '/_data'

        if self.name == 'mnist':
            func = torchvision.datasets.MNIST

        if self.name == 'mnistf':
            func = torchvision.datasets.FashionMNIST

        if self.name == 'cifar10':
            func = torchvision.datasets.CIFAR10

        if self.name == 'imagenet':
            func = None

        if func:
            self.data_trn = func(root=fpath, train=True, download=True,
                transform=self.transform)

            self.data_tst = func(root=fpath, train=False, download=True,
                transform=self.transform)

            self.dataloader_trn = DataLoader(self.data_trn,
                batch_size=self.batch_trn, shuffle=True)
            self.dataloader_tst = DataLoader(self.data_tst,
                batch_size=self.batch_tst, shuffle=True)

        else:
            self.data_trn = None
            self.data_tst = None

            self.dataloader_trn = None
            self.dataloader_tst = None

    def load_labels(self):
        self.labels = {}

        if self.name == 'mnist':
            self.labels = {
                0: '0',
                1: '1',
                2: '2',
                3: '3',
                4: '4',
                5: '5',
                6: '6',
                7: '7',
                8: '8',
                9: '9',
            }

        if self.name == 'mnistf':
            self.labels = {
                0: 'T-Shirt',
                1: 'Trouser',
                2: 'Pullover',
                3: 'Dress',
                4: 'Coat',
                5: 'Sandal',
                6: 'Shirt',
                7: 'Sneaker',
                8: 'Bag',
                9: 'Ankle Boot',
            }

        if self.name == 'cifar10':
            self.labels = {
                0: 'airplane',
                1: 'automobile',
                2: 'bird',
                3: 'cat',
                4: 'deer',
                5: 'dog',
                6: 'frog',
                7: 'horse',
                8: 'ship',
                9: 'truck',
            }

        if self.name == 'imagenet':
            IMAGENET_URL = 'https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt'
            labels = ''
            for f in urllib.request.urlopen(IMAGENET_URL):
                labels = labels + f.decode('utf-8')

            self.labels = yaml.safe_load(labels)

    def load_shape(self):
        self.sz = 0
        self.ch = 0

        if self.name == 'mnist':
            self.sz = 28
            self.ch = 1

        if self.name == 'mnistf':
            self.sz = 28
            self.ch = 1

        if self.name == 'cifar10':
            self.sz = 32
            self.ch = 3

        if self.name == 'imagenet':
            self.sz = 224
            self.ch = 3

    def load_transform(self):
        self.transform = None

        if self.name == 'mnist':
            self.transform = torchvision.transforms.ToTensor()

        if self.name == 'mnistf':
            self.transform = torchvision.transforms.ToTensor()

        if self.name == 'cifar10':
            # TODO: add support for SNN selection (0.5)
            m = (0.4914, 0.4822, 0.4465)
            v = (0.2471, 0.2435, 0.2616)
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(m, v),
            ])

        if self.name == 'imagenet':
            m = [0.485, 0.456, 0.406]
            v = [0.229, 0.224, 0.225]
            self.transform = torchvision.transforms.Compose([
                # torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(self.sz),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(m, v),
                # torchvision.transforms.Lambda(lambda x: x[None]),
            ])

    def plot(self, x, title='', fpath=None):
        x = x.detach().to('cpu')
        size = 8 if self.name == 'imagenet' else 3

        fig = plt.figure(figsize=(size, size))

        if self.ch == 1:
            img = x.squeeze()
            plt.imshow(img, cmap='gray')

        elif self.name == 'imagenet':
            img = self.tensor_to_plot(x)
            plt.imshow(img)

        else:
            img = torchvision.utils.make_grid(x, nrow=1, normalize=True)
            img = img.transpose(0, 2).transpose(0, 1)
            plt.imshow(img)

        plt.title(title)
        plt.axis('off')

        if fpath:
            plt.savefig(fpath, bbox_inches='tight')
        else:
            plt.show()

    def plot_many(self, X=None, titles=None, cols=3, rows=3, fpath=None):
        fig = plt.figure(figsize=(3*cols, 3*rows))

        for j in range(1, cols * rows + 1):
            if X is None:
                i = torch.randint(len(self.data_trn), size=(1,)).item()
                x, c = self.data_trn[i]
                title = self.labels.get(c, '')
            else:
                x = X[j-1].detach().to('cpu')
                title = titles[j-1] if titles else ''

            fig.add_subplot(rows, cols, j)
            plt.title(title)
            plt.axis('off')

            if self.ch == 1:
                img = x.squeeze()
                plt.imshow(img, cmap='gray')

            elif self.name == 'imagenet':
                img = self.tensor_to_plot(x)
                plt.imshow(img)

            else:
                img = torchvision.utils.make_grid(x, nrow=1, normalize=True)
                img = img.transpose(0, 2).transpose(0, 1)
                plt.imshow(img)

        if fpath:
            plt.savefig(fpath, bbox_inches='tight')
        else:
            plt.show()

    def tensor_to_plot(self, x):
        """Transform tensor to image for imagenet-like data."""
        if torch.is_tensor(x):
            x = x.detach().to('cpu').numpy()
        x = x.transpose((1, 2, 0))
        m = np.array([0.4451, 0.4262, 0.3959])
        s = np.array([0.2411, 0.2403, 0.2466])
        x = s * x + m
        x = np.clip(x, 0, 1)
        return x
