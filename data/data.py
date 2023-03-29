from PIL import Image
from matplotlib.animation import FuncAnimation
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

    def animate(self, X, titles, fpath=None):
        if self.name != 'cifar10':
            raise NotImplementedError('It works now only for cifar10')

        if X is None or len(X) == 0 or len(X) != len(titles):
            print('WRN: invalid data for animation')
            return

        def prep(x):
            x = x.detach().to('cpu')
            x = torchvision.utils.make_grid(x, nrow=1, normalize=True)
            x = x.transpose(0, 2).transpose(0, 1)
            x = x.numpy()
            return x

        fig = plt.figure(figsize=(3, 3))
        ax = fig.add_subplot(111)
        ax.axis('off')

        img = ax.imshow(prep(X[0]))

        def update(k, *args):
            ax.set_title(titles[k], fontsize=10)
            img.set_data(prep(X[k]))
            return (img,)

        anim = FuncAnimation(fig, update, interval=10,
            frames=len(X), blit=True, repeat=False)

        anim.save(fpath, writer='pillow', fps=0.7) if fpath else anim.show()
        plt.close(fig)

    def get(self, i=None, tst=False):
        data = self.data_tst if tst else self.data_trn

        if i is None:
            i = torch.randint(len(data), size=(1,)).item()
        else:
            i = min(i, len(data)-1)

        x, c = data[i]
        l = self.labels.get(c)

        return x, c, l

    def img_load(self, fpath, device='cpu', wo_norm=False):
        img = Image.open(fpath)
        transform = self.transform_wo_norm if wo_norm else self.transform
        return transform(img).to(device)

    def img_rand(self, device='cpu', wo_norm=False):
        pix = np.random.rand(self.sz, self.sz, self.ch) * 255
        img = Image.fromarray(pix.astype('uint8')).convert('RGB')
        transform = self.transform_wo_norm if wo_norm else self.transform
        return transform(img).to(device)

    def load_data(self):
        fpath = os.path.dirname(__file__) + '/_data'

        if self.name == 'mnist':
            func = torchvision.datasets.MNIST
            load = not os.path.isdir(os.path.join(fpath, 'MNIST'))

        if self.name == 'mnistf':
            func = torchvision.datasets.FashionMNIST
            load = not os.path.isdir(os.path.join(fpath, 'FashionMNIST'))

        if self.name == 'cifar10':
            func = torchvision.datasets.CIFAR10
            load = not os.path.isdir(os.path.join(fpath, 'cifar-10-batches-py'))

        if self.name == 'imagenet':
            func = None

        if func:
            self.data_trn = func(root=fpath, train=True, download=load,
                transform=self.transform)

            self.data_tst = func(root=fpath, train=False, download=load,
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
                0: 't-shirt',
                1: 'trouser',
                2: 'pullover',
                3: 'dress',
                4: 'coat',
                5: 'sandal',
                6: 'shirt',
                7: 'sneaker',
                8: 'bag',
                9: 'ankle boot',
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
            # (the current m/v are for densenet)
            m = (0.4914, 0.4822, 0.4465)
            v = (0.2471, 0.2435, 0.2616)
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(m, v),
            ])

        self.transform_wo_norm = torchvision.transforms.ToTensor()

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
            self.transform_wo_norm = torchvision.transforms.Compose([
                torchvision.transforms.CenterCrop(self.sz),
                torchvision.transforms.ToTensor(),
            ])

    def plot(self, x, title='', fpath=None, is_new=True):
        size = 8 if self.name == 'imagenet' else 3
        cmap = 'hot' if self.name in ['mnist', 'mnistf'] else None

        if self.name in ['cifar10']:
            x = self.tensor_to_plot_cifar10(x)
        if self.name in ['imagenet']:
            x = self.tensor_to_plot_imagenet(x)

        return self.plot_base(x, title, size, cmap, fpath, is_new)

    def plot_base(self, x, title, size=3, cmap='hot', fpath=None, is_new=True):
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        x = x.detach().to('cpu').squeeze()

        if is_new:
            fig = plt.figure(figsize=(size, size))

        plt.imshow(x, cmap=cmap)
        plt.title(title)
        plt.axis('off')

        if fpath:
            plt.savefig(fpath, bbox_inches='tight')
        elif is_new:
            plt.show()
            plt.close(fig)

    def plot_many(self, X=None, titles=None, cols=3, rows=3, size=3, fpath=None):
        fig = plt.figure(figsize=(size*cols, size*rows))

        for j in range(1, cols * rows + 1):
            if X is None:
                i = torch.randint(len(self.data_trn), size=(1,)).item()
                x, c, l = self.get(i)
                title = l
            else:
                x = X[j-1].detach().to('cpu')
                title = titles[j-1] if titles else ''

            fig.add_subplot(rows, cols, j)
            self.plot(x, title, is_new=False)

        plt.savefig(fpath, bbox_inches='tight') if fpath else plt.show()
        plt.close(fig)

    def tensor_to_plot_cifar10(self, x):
        """Transform tensor to image for cifar10-like data."""
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        x = x.detach().to('cpu')
        x = torchvision.utils.make_grid(x, nrow=1, normalize=True)
        x = x.transpose(0, 2).transpose(0, 1)
        return x

    def tensor_to_plot_imagenet(self, x):
        """Transform tensor to image for imagenet-like data."""
        if torch.is_tensor(x):
            x = x.detach().to('cpu').numpy()
        x = x.transpose((1, 2, 0))
        m = np.array([0.4451, 0.4262, 0.3959])
        s = np.array([0.2411, 0.2403, 0.2466])
        x = s * x + m
        x = np.clip(x, 0, 1)
        return torch.tensor(x)
