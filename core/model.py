import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from time import perf_counter as tpc
import torch
import torchvision


class Model:
    def __init__(self, name, device, sz, ch, labels=None):
        self.name = name
        self.device = device

        self.sz = sz
        self.ch = ch
        self.labels = labels

        self.ann = torch.hub.load('pytorch/vision', name, weights=True)
        self.ann.to(self.device)

        self.probs = torch.nn.Softmax(dim=1)

        self.rmv_target(is_init=True)

    def get_a(self):
        # Return activation of target neuron as a number
        a = self.hook.a_mean.detach().to('cpu').numpy()
        return float(a)

    def img_load(self, fpath):
        img = Image.open(fpath)
        return self.img_to_tensor(img)

    def img_rand(self):
        pix = np.random.rand(self.sz, self.sz, self.ch) * 255
        img = Image.fromarray(pix.astype('uint8')).convert('RGB')
        return self.img_to_tensor(img)

    def img_show(self, x, title='', fpath=None, with_trans=True):
        fig = plt.figure(figsize=(6, 6))
        plt.imshow(self.tensor_to_plot(x))
        plt.axis('off')
        plt.title(title)

        if fpath is None:
            plt.show()
        else:
            plt.savefig(fpath, bbox_inches='tight')

    def tensor_to_plot(self, x):
        x = x.detach().to('cpu').numpy()
        x = x.transpose((1, 2, 0))
        m = np.array([0.4451, 0.4262, 0.3959])
        s = np.array([0.2411, 0.2403, 0.2466])
        x = s * x + m
        x = np.clip(x, 0, 1)
        return x

    def img_to_tensor(self, img):
        m = [0.485, 0.456, 0.406]
        v = [0.229, 0.224, 0.225]

        tens = torchvision.transforms.Compose([
            # torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(self.sz),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(m, v),
            # torchvision.transforms.Lambda(lambda x: x[None]),
        ])(img)

        return tens.to(self.device)

    def rmv_target(self, is_init=False):
        if is_init:
            self.hook_hand = []
        else:
            while len(self.hook_hand) > 0:
                self.hook_hand.pop().remove()
            self.hook_hand = []

        self.layer = None
        self.filter = None
        self.hook = None

    def run(self, x):
        is_batch = len(x.shape) == 4
        if not is_batch:
            x = x[None]

        with torch.no_grad():
            self.ann.eval()
            y = self.ann(x)
            y = self.probs(y)

        return y if is_batch else y[0]

    def run_target(self, x):
        is_batch = len(x.shape) == 4

        self.run(x)

        return self.hook.a if is_batch else self.hook.a_mean

    def set_target(self, layer, filter):
        self.layer = self.ann.features[layer]
        if type(self.layer) != torch.nn.modules.conv.Conv2d:
            raise ValueError('We work only with conv layers')

        self.filter = filter
        if self.filter < 0 or self.filter >= self.layer.out_channels:
            raise ValueError('Filter does not exist')

        self.hook = AmHook(self.filter)
        self.hook_hand = [self.layer.register_forward_hook(self.hook.forward)]


class AmHook():
    def __init__(self, filter):
        self.filter = filter
        self.a = None
        self.a_mean = None

    def forward(self, module, inp, out):
        self.a = torch.mean(out[:, self.filter, :, :], dim=(1, 2))
        self.a_mean = torch.mean(out[:, self.filter, :, :])
