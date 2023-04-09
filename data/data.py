from PIL import Image
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
import torchvision


from .data_opts import DATA_OPTS


class Data:
    def __init__(self, name, batch_trn=256, batch_tst=32, norm_m=None, norm_v=None):
        if not name in DATA_OPTS.keys():
            raise ValueError(f'Dataset name "{name}" is not supported')
        self.name = name

        self.batch_trn = batch_trn
        self.batch_tst = batch_tst

        self.opts = DATA_OPTS[self.name]
        self.labels = self.opts['labels']
        self.sz = self.opts['sz']
        self.ch = self.opts['ch']

        self.load_transform(norm_m, norm_v)
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
            ax.set_title(titles[k], fontsize=9)
            img.set_data(prep(X[k]))
            return (img,)

        anim = FuncAnimation(fig, update, interval=6,
            frames=len(X), blit=True, repeat=False)

        anim.save(fpath, writer='pillow', fps=0.7) if fpath else anim.show()
        plt.close(fig)

    def get(self, i=None, tst=False):
        data = self.data_tst if tst else self.data_trn

        if i is None:
            i = torch.randint(len(data), size=(1,)).item()

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

    def info(self):
        text = ''
        text += f'Dataset             : {self.name}\n'
        text += f'Number of classes   : {len(self.labels):-10d}\n'

        if self.data_trn is not None:
            text += f'Size of trn dataset : {len(self.data_trn):-10d}\n'
        if self.data_tst is not None:
            text += f'Size of tst dataset : {len(self.data_tst):-10d}\n'
        if self.var_trn is not None:
            text += f'Var  of trn dataset : {self.var_trn:-10.4e}\n'
        if self.var_tst is not None:
            text += f'Var  of tst dataset : {self.var_tst:-10.4e}\n'

        return text

    def load_data(self):
        self.data_trn = None
        self.data_tst = None
        self.dataloader_trn = None
        self.dataloader_tst = None
        self.var_trn = None
        self.var_tst = None

        if self.opts['dataset'] is not None:
            func = eval(f'torchvision.datasets.{self.opts["dataset"]}')
            fpath = os.path.dirname(__file__) + '/_data'
            fpath_check = os.path.join(fpath, self.opts['dataset_check'])
            download = not os.path.isdir(fpath_check)

            self.data_trn = func(root=fpath, train=True, download=download,
                transform=self.transform)
            self.data_tst = func(root=fpath, train=False, download=download,
                transform=self.transform)

            self.dataloader_trn = DataLoader(self.data_trn,
                batch_size=self.batch_trn, shuffle=True)
            self.dataloader_tst = DataLoader(self.data_tst,
                batch_size=self.batch_tst, shuffle=True)

            self.var_trn = np.var(self.data_trn.data / 255.0) # TODO: check
            self.var_tst = np.var(self.data_tst.data / 255.0)

    def load_transform(self, norm_m=None, norm_v=None):
        norm_m = norm_m or self.opts.get('norm_m')
        norm_v = norm_v or self.opts.get('norm_v')
        if norm_m is not None and norm_v is not None:
            transform_norm = torchvision.transforms.Normalize(norm_m, norm_v)
            self.transform_norm = torchvision.transforms.Compose([
                transform_norm,
            ])
        else:
            transform_norm = None
            self.transform_norm = None

        self.transform = torchvision.transforms.ToTensor()
        self.transform_wo_norm = torchvision.transforms.ToTensor()

        if self.name == 'cifar10':
            self.transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                transform_norm,
            ])

        if self.name == 'imagenet':
            self.transform = torchvision.transforms.Compose([
                #torchvision.transforms.Resize(256),
                #torchvision.transforms.CenterCrop(self.sz),
                torchvision.transforms.ToTensor(),
                transform_norm,
            ])
            self.transform_wo_norm = torchvision.transforms.Compose([
                #torchvision.transforms.Resize(256),
                #torchvision.transforms.CenterCrop(self.sz),
                torchvision.transforms.ToTensor(),
            ])

    def plot(self, x, title='', fpath=None, is_new=True):
        if self.name in ['cifar10']:
            x = self.tensor_to_plot_cifar10(x)
        if self.name in ['imagenet']:
            x = self.tensor_to_plot_imagenet(x)

        return self.plot_base(x, title,
            self.opts['plot_size'], self.opts['plot_cmap'], fpath, is_new)

    def plot_base(self, x, title, size=3, cmap='hot', fpath=None, is_new=True):
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        x = x.detach().to('cpu').squeeze()

        if is_new:
            fig = plt.figure(figsize=(size, size))

        plt.imshow(x, cmap=cmap)
        plt.title(title, fontsize=9)
        plt.axis('off')

        if fpath:
            plt.savefig(fpath, bbox_inches='tight')
        elif is_new:
            plt.show()
            plt.close(fig)

    def plot_many(self, X=None, titles=None, cols=5, rows=5, size=3, fpath=None):
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
