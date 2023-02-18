import itertools
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torchvision
import urllib
import yaml


class Data:
    def __init__(self, name):
        self.name = name

        self.load_shape()

    @property
    def func(self):
        if self.name == 'mnist':
            return torchvision.datasets.MNIST

        if self.name == 'mnistf':
            return torchvision.datasets.FashionMNIST

        if self.name == 'imagenet':
            return None

        raise ValuerError('Invalid data name')

    def get(self, i=None, tst=False):
        data = self.data_tst if tst else self.data_trn

        if i is None:
            i = torch.randint(len(data), size=(1,)).item()
        else:
            i = min(i, len(data)-1)

        X, c = data[i]
        return X, c

    def load(self, fpath='result/_data', batch_size=64):
        if not self.func:
            return

        self.data_trn = self.func(
            root=fpath, train=True, download=True,
            transform=torchvision.transforms.ToTensor())

        self.data_tst = self.func(
            root=fpath, train=False, download=True,
            transform=torchvision.transforms.ToTensor())

        self.dataloader_trn = DataLoader(self.data_trn, batch_size=batch_size,
            shuffle=True)
        self.dataloader_tst = DataLoader(self.data_tst, batch_size=batch_size,
            shuffle=True)

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

        if self.name == 'imagenet':
            self.sz = 224
            self.ch = 3

    def show(self, fpath=None):
        fig = plt.figure(figsize=(8, 8))
        cols, rows = 3, 3
        for i in range(1, cols * rows + 1):
            sample_idx = torch.randint(len(self.data_trn), size=(1,)).item()
            img, label = self.data_trn[sample_idx]
            fig.add_subplot(rows, cols, i)
            plt.title(self.labels.get(label, ''))
            plt.axis('off')
            if self.ch == 1:
                plt.imshow(img.squeeze(), cmap='gray')
            else:
                raise NotImplementedError('It works only for grayscale images')

        if fpath:
            plt.savefig(fpath, bbox_inches='tight')
        else:
            plt.show()
