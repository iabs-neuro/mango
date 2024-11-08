from contextlib import nullcontext
from collections import OrderedDict

from spikingjelly.activation_based import functional
from spikingjelly.activation_based.functional import reset_net
import torch.nn.functional as F

import numpy as np
import os
import torch
import warnings

from .densenet_cifar10.densenet_cifar10 import DensenetCifar10
from .snn_cifar10.snn_cifar10 import SNNCifar10
from .snn_cifar10.sj_snn_cifar10 import SJSNNCifar10
from .resnet_cifar10.resnet import ResNet18
from ..utils import load_yandex

# To remove the warning of torchvision:
warnings.filterwarnings('ignore', category=UserWarning)

NAMES = ['alexnet', 'densenet', 'vgg16', 'vgg19', 'snn', 'sjsnn', 'resnet18']

class Model:
    def __init__(self, name, data, device='cpu', model_path=None):
        if name not in NAMES:
            raise ValueError(f'Model name "{name}" is not supported')
        self.name = name
        self.data = data
        self.is_snn = ('snn' in self.name)

        self.device = device
        self.probs = torch.nn.Softmax(dim=1)
        self.model_path = model_path
        self.load()

        self.rmv_target(is_init=True)
        self.target_mode = None

    def x_to_image(self, x):
        xt = torch.tensor(x)
        res = self.data.tr_norm_inv(xt).detach().cpu().squeeze().numpy()
        if len(res.shape) == 3:
            res = res.transpose(1, 2, 0)
            res = np.clip(res, 0, 1) if np.mean(res) < 2 else np.clip(res, 0, 255)
        # res = np.uint8(np.moveaxis(res.detach().squeeze().numpy(), 0, 2) * 256)
        return res

    def attrib(self, x, c=None, steps=3, iters=10):
        if c is None:
            y, c, l = self.run_pred(x)

        x = self.data.tr_norm_inv(x)
        x = np.uint8(np.moveaxis(x.numpy(), 0, 2) * 256)

        def _img_to_x(x):
            # m = np.array([0.485, 0.456, 0.406]).reshape([1, 1, 3])
            # s = np.array([0.229, 0.224, 0.225]).reshape([1, 1, 3])
            # x = (x / 255 - m) / s
            x = np.transpose(x, (2, 0, 1)) / 255
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
            x = self.data.tr_norm(x).unsqueeze(0)
            # x = np.expand_dims(x, 0)
            # x = np.array(x)
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
            return x

        def _iter(x):
            x = _img_to_x(x)
            x.requires_grad_()
            y = self.probs(self.net(x))[0, c]
            self.net.zero_grad()
            y.backward()
            return x.grad.detach().cpu().numpy()[0]

        def _thr(a, p=60):
            if p >= 100:
                return np.min(a)
            a_sort = np.sort(np.abs(a.flatten()))[::-1]
            s = 100. * np.cumsum(a_sort) / np.sum(a)
            i = np.where(s >= p)[0]
            return a_sort[i[0]]

        ig = []
        for _ in range(iters):
            x0 = 255. * np.random.random(x.shape)
            xs = [x0 + 1. * i / steps * (x - x0) for i in range(steps)]

            g = [_iter(x_) for x_ in xs]
            g_avg = np.average(g, axis=0)
            g_avg = np.transpose(g_avg, (1, 2, 0))

            x_delta = _img_to_x(x) - _img_to_x(x0)
            x_delta = x_delta.detach().squeeze(0).cpu().numpy()
            x_delta = np.transpose(x_delta, (1, 2, 0))

            ig.append(x_delta * g_avg)

        a = np.average(np.array(ig), axis=0)

        a = np.average(np.clip(a, 0., 1.), axis=2)
        m = _thr(a, 1)
        e = _thr(a, 100)
        a_thr = (np.abs(a) - e) / (m - e)
        a_thr *= np.sign(a)
        a_thr *= (a_thr >= 0.)
        x = np.expand_dims(np.clip(a_thr, 0., 1.), 2) * [0, 255, 0]

        x = np.moveaxis(x, 2, 0)
        x = 0.2989 * x[0, :, :] + 0.5870 * x[1, :, :] + 0.1140 * x[2, :, :]
        return x / np.max(x)

    def check(self, tst=True, only_one_batch=False, with_target=False):
        data = self.data.dataloader_tst if tst else self.data.dataloader_trn
        n, m, a = 0, 0, []

        for x, l_real in data:
            x = x.to(self.device)
            y = self.run(x)
            l = torch.argmax(y, dim=1).detach().to('cpu')
            m += (l == l_real).sum()
            n += len(l)

            if with_target:
                a_cur = self.run_target(x)
                a.extend(list(a_cur.detach().to('cpu').numpy()))

            if only_one_batch:
                break

        return (n, m, np.array(a)) if with_target else (n, m)

    def load(self):
        fpath = os.path.dirname(__file__)
        os.makedirs(fpath, exist_ok=True)

        self.net = None

        if self.name == 'densenet':
            if self.data.name != 'cifar10':
                msg = 'Model "densenet" is ready only for "cifar10"'
                raise NotImplementedError(msg)

            fpath = os.path.join(fpath, 'densenet_cifar10', 'densenet_cifar10.pt')

            if not os.path.isfile(fpath):
                load_yandex('https://disk.yandex.ru/d/ndE0NjV2G72skw', fpath)

            self.net = DensenetCifar10()
            state_dict = torch.load(fpath, map_location=self.device)
            self.net.load_state_dict(state_dict)

        if self.name in ['alexnet', 'vgg16', 'vgg19']:
            if self.data.name != 'imagenet':
                msg = f'Model "{self.name}" is ready only for "imagenet"'
                raise NotImplementedError(msg)

            # TODO: set path to data

            self.net = torch.hub.load('pytorch/vision:v0.10.0',
                                      self.name,
                                      weights=True)

        if 'resnet' in self.name:

            if self.model_path is None:
                fpath = os.path.join(fpath, 'resnet_cifar10', 'resnet18_cifar10_best.pth')
            else:
                fpath = self.model_path

            self.net = ResNet18()
            state_dict = torch.load(fpath, map_location=self.device)
            net_dict = state_dict['net']
            renamed_state_dict = OrderedDict()
            for i, pair in enumerate(net_dict.items()):
                new_name = pair[0].split('module.')[1]
                renamed_state_dict[new_name] = net_dict[pair[0]]
            self.net.load_state_dict(renamed_state_dict, strict=True)

        if self.name == 'snn':
            if self.data.name != 'cifar10':
                msg = f'Model "{self.name}" is ready only for "cifar10"'
                raise NotImplementedError(msg)

            fpath = os.path.join(fpath, 'snn_cifar10', 'snn_cifar10.pt')

            self.net = SNNCifar10()
            state_dict = torch.load(fpath, map_location=self.device)
            renamed_state_dict = OrderedDict()
            for i, pair in enumerate(state_dict.items()):
                new_name = 'features.' + pair[0]
                renamed_state_dict[new_name] = state_dict[pair[0]]
            self.net.load_state_dict(renamed_state_dict, strict=True)

        if self.name == 'sjsnn':
            if self.data.name != 'cifar10':
                msg = f'Model "{self.name}" is ready only for "cifar10"'
                raise NotImplementedError(msg)

            if self.model_path is None:
                fpath = os.path.join(fpath, 'snn_cifar10', 'checkpoint_max_test_acc1_t50.pth')
            else:
                fpath = self.model_path

            self.net = SJSNNCifar10()
            functional.set_step_mode(self.net, step_mode='m')
            functional.set_backend(self.net, 'cupy', self.net.spiking_neuron) # sets super-fast cupy backend

            checkpoint = torch.load(fpath, map_location=self.device)
            state_dict = checkpoint['model']
            self.net.load_state_dict(state_dict, strict=True)

        if self.net is not None:
            self.net.to(self.device)
            self.net.eval()

    def get_a(self):
        # Return activation of target neuron as a number
        a = self.hook.a_mean.detach().to('cpu').numpy()
        return float(a)

    def has_target(self):
        return (self.cls is not None) or (self.layer is not None and self.unit is not None)

    def rmv_target(self, is_init=False):
        if is_init:
            self.hook_hand = []
        else:
            while len(self.hook_hand) > 0:
                self.hook_hand.pop().remove()
            self.hook_hand = []

        self.cls = None
        self.layer = None
        self.unit = None
        self.hook = None


    def _preprocess_input(self, x):
        is_batch = (len(x.shape) == 4)
        if not is_batch:
            x = x[None]

        return x, is_batch

    def run(self, x, with_grad=False):
        x, is_batch = self._preprocess_input(x)
        x = x.to(self.device)

        with nullcontext() if with_grad else torch.no_grad():
            y = self.net(x)
            if self.is_snn:
                y = torch.mean(y, dim=0)  # sum over timeframes
            else:
                y = self.probs(y)

        #y = self.probs(y)
        return y if is_batch else y[0]

    def run_pred(self, x):
        x, is_batch = self._preprocess_input(x)

        if self.name == 'sjsnn':  # reset neuron internal structure for inference
            reset_net(self.net)

        y = self.run(x).detach().to('cpu').numpy()

        c = np.argmax(y, axis=1)
        p = np.array([y[i, c_cur] for i, c_cur in enumerate(c)])
        l = [self.data.labels[c_cur] for c_cur in c]

        return (p, c, l) if is_batch else (p[0], c[0], l[0])

    def run_target(self, x):
        '''
        x has shape [k, ch, sz, sz], k=number of samples from optimization algorithm
        '''
        x, is_batch = self._preprocess_input(x)

        if self.name == 'sjsnn':  # reset neuron internal structure for inference
            reset_net(self.net)

        y = self.run(x)

        #print('y:', y)
        #print(self.hook.__dict__)
        #print(self.hook_result)

        if self.cls is not None:
            res = y[:, self.cls]
        else:
            res = self.hook.a
            #print(res.shape)

        return res if is_batch else res[0]

    def set_target_mode(self, cls=None, layer=None, unit=None):
        if cls is not None:
            if layer is not None or unit is not None:
                raise ValueError('Please, set class or layer + unit, not both')
            else:
                self.target_mode = 'class'

        elif layer is None or unit is None:
            raise ValueError('Class was not set, dispatcher needs both layer + unit to fix target,'\
                             f'but got layer={layer} and unit={unit}')

        else:
            self.target_mode = 'unit'

    def _get_layer(self, layer, logger=None):
        if isinstance(layer, str):
            # name of layer provided
            if self.name == 'sjsnn':
                print(list(self.net.named_layers_od.keys()))
                try:
                    model_layer = self.net.named_layers_od[layer]
                except KeyError:
                    msg = f'{layer} not found in model layers. Select one of: {list(self.net.named_layers_od.keys())}'
                    if logger is not None:
                        logger(msg)
                    raise ValueError(msg)

            else:
                try:
                    #print(self.net._modules)
                    if 'resnet' in self.name: # here we search for the target layer hierarchically
                        layer_hierarchy = layer.split('.')
                        current_level = self.net._modules
                        for lh in layer_hierarchy:
                            if isinstance(current_level, torch.nn.Sequential):
                                try:
                                    lh = int(lh)
                                except:
                                    raise ValueError(f'Cannot define the layer position inside nn.Sequential with a string {lh}.\
                                     The layer id must be convertable to int, consider renaming or check layer structure')

                            try:
                                current_level = current_level[lh]
                            except:
                                current_level = getattr(current_level, lh)

                        model_layer = current_level
                        print(f'selected layer: {model_layer}')

                    if self.name == 'densenet':
                        model_layer = getattr(self.net._modules['features'], f'{layer}')

                except KeyError:
                    msg = f'{layer} not found in model layers. Select one of: {list(self.net.named_modules.keys())}'
                    if logger is not None:
                        logger(msg)
                    raise ValueError(msg)

        else:
            # index of layer provided
            try:
                model_layer = self.net.features[int(layer)]
            except AttributeError:  # exception for SJ SNN
                model_layer = list(self.net.named_layers_od.items())[int(layer)][1]

        return model_layer

    def set_target(self, cls=None, layer=None, unit=None, logger=None):
        self.cls = None
        self.layer = None
        self.unit = None

        if self.target_mode is not None:
            self.set_target_mode(cls, layer, unit)

        if self.target_mode == 'class':
            self.cls = int(cls)
            if logger is not None:
                logger(f'Target set for class {self.cls}')
            return

        elif self.target_mode == 'unit':
            self.layer = self._get_layer(layer, logger=logger)
            self.unit = int(unit)

            if logger is not None:
                logger(f'Target set for layer {self.layer} and unit {self.unit}')

            '''
            if type(layer) != torch.nn.modules.conv.Conv2d:
                raise ValueError('We work only with conv layers')
            
            if self.f < 0 or self.f >= layer.out_channels:
                raise ValueError('Filter does not exist')
            '''
            self.hook = AmHook(self, self.unit)
            self.hook_hand = [self.layer.register_forward_hook(self.hook.forward)]
            self.hook_result = []

    def set_activity_hooks(self, layers=None, logger=None):
        self.hooks = {}
        for l in layers:
            hook = LayerHook(self)
            net_layer = self._get_layer(l, logger=logger)
            self.hooks[l] = net_layer.register_forward_hook(hook.forward)
        self.hook_result = []

class LayerHook():
    def __init__(self, parent_model):
        self.parent_model = parent_model
        self.a = None
        self.a_mean = None
        self.shape_shown = False  # TODO: technical field for debug, remove later

    def forward(self, module, inp, out):
        if not self.shape_shown:
            print('hook input:', inp[0].shape)
            #print('hook output:', out)
            print('hook shape', out.shape)
            self.shape_shown = True

        if self.parent_model.name == 'sjsnn':  # spikingjelly supports [T,N,U,W,H] format (one extra dimension)
            if len(out.shape) == 5:
                self.a = torch.mean(out[:, :, :, :, :], dim=(3, 4))
                self.a_mean = torch.mean(out[:, :, :, :, :])
                # print('hook:', self.parent_model.hook_result)
                self.parent_model.hook_result.append(self.a)
            elif len(out.shape) == 3:
                self.a = out[:, :, :]
                self.a_mean = torch.mean(out[:, :, :])
                self.parent_model.hook_result.append(self.a)

        else:
            if len(out.shape) == 2:
                self.a = out[:, :]
                self.a_mean = None
                self.parent_model.hook_result.append(self.a)
            else:
                self.a = torch.mean(out[:, :, :, :], dim=(1, 2))
                self.a_mean = torch.mean(out[:, :, :, :])
                #print('hook:', self.parent_model.hook_result)
                self.parent_model.hook_result.append(self.a)


class AmHook():
    def __init__(self, parent_model, unit):
        self.parent_model = parent_model
        self.unit = unit
        self.a = None
        self.a_mean = None
        self.shape_shown = False  # TODO: technical field for debug, remove later

    def forward(self, module, inp, out):
        if not self.shape_shown:
            print('hook input:', inp[0].shape)
            #print('hook output:', out)
            print('hook shape', out.shape)
            try:
                print(inp[0][0, 0, self.unit, :, :])
                print(out[0, 0, self.unit, :, :])
            except:
                pass
            self.shape_shown = True

        if self.parent_model.name == 'sjsnn':  # spikingjelly supports [T,N,C,W,H] format (one extra dimension)
            if len(out.shape) == 5:
                self.a = torch.mean(out[:, :, self.unit, :, :], dim=(0, 2, 3))
                self.a_mean = torch.mean(out[:, :, self.unit, :, :])
                # print('hook:', self.parent_model.hook_result)
                self.parent_model.hook_result.append(self.a)
            elif len(out.shape) == 3:
                self.a = torch.mean(out[:, :, self.unit], dim=0)
                self.a_mean = torch.mean(out[:, :, self.unit])
                self.parent_model.hook_result.append(self.a)

        elif 'resnet' in self.parent_model.name: # for ANN the activity of units in conv layer is targeted
            if len(out.shape) == 4:
                self.a = torch.mean(F.relu(out[:, self.unit, :, :]), dim=(1, 2)) # add Relu as non-linearity manually
                self.a_mean = torch.mean(F.relu(out[:, self.unit, :, :]))
                self.parent_model.hook_result.append(self.a)
            else:
                raise ValueError('Wrong tensor format')

        else:
            if len(out.shape) == 2:
                self.a = out[:, self.unit]
                self.a_mean = None
                self.parent_model.hook_result.append(self.a)
            else:
                self.a = torch.mean(out[:, self.unit, :, :], dim=(1, 2))
                self.a_mean = torch.mean(out[:, self.unit, :, :])
                #print('hook:', self.parent_model.hook_result)
                self.parent_model.hook_result.append(self.a)
