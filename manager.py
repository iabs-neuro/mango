import numpy as np
import os
import random
import sys
from time import perf_counter as tpc
import torch


from data import Data
from gen import Gen
from model import Model
from opt import opt_protes
from utils import Log


from protes import protes


NAMES = ['data', 'densenet_out']


class Manager:
    def __init__(self, name, n=64, lim_a=-4., lim_b=+4., device=None):
        if not name in NAMES:
            raise ValueError(f'Manager name "{name}" is not supported')

        self.name = name    # Task name
        self.n = n          # Grid size
        self.lim_a = lim_a  # Grid lower limit
        self.lim_b = lim_b  # Grid upper limit

        self.set_rand()
        self.set_device(device)
        self.set_path()
        self.set_log()
        self.load()
        eval('self.run_' + self.name + '()')

    def func(self, z_index, with_score=False):
        z = self.ind_to_poi(z_index)
        x = self.gen.run(z)
        s = self.gen.score(x).detach().to('cpu').numpy() if with_score else 0.
        a = self.model.run_target(x).detach().to('cpu').numpy()
        return a + 0.1 * s

    def func_ind(self, z_index, with_score=False):
        return self.func(self.ind_to_poi(z_index), with_score)

    def get_path(self, fpath):
        return os.path.join(self.path_result, fpath)

    def ind_to_poi(self, z_index):
        z_index = np.array(z_index) # From jax to numpy
        z = z_index / (self.n - 1) * (self.lim_b - self.lim_a) + self.lim_a
        z = torch.tensor(z, dtype=torch.float32, device=self.device)
        return z

    def load(self):
        if self.name in ['data']:
            self.data = {}
            for name_data in ['mnist', 'mnistf', 'cifar10', 'imagenet']:
                tm = self.log.prc(f'Loading "{name_data}" dataset')
                self.data[name_data] = Data(name_data)
                self.log.res(tpc()-tm)

        if self.name in ['densenet_out']:
            self.data = Data('cifar10')
            self.model = Model('densenet161', self.data, self.device)
            self.gen = Gen('gan-sn', self.device)

    def run_densenet_out(self, evals=1.E+4):
        X, titles = [], []
        for cl in np.arange(10):
            print(f'\n>>> Optimize class {cl} ({self.data.labels[cl]}) : ')
            self.model.set_target(cl=cl)

            info = {}
            z_index = protes(self.func, self.gen.d, self.n, int(evals),
                is_max=True, log=True, info=info, with_info_i_opt_list=True,
                k=10, k_top=2)[0]
            z = self.ind_to_poi(z_index)
            x = self.gen.run(z)
            y = self.model.run_target(x)
            title = f'y={y:-8.2e} : {self.data.labels[cl]}'
            self.data.plot(x, title,
                fpath=self.get_path(f'img/out_max_protes_class{cl}.png'))
            X.append(x)
            titles.append(title)

            X_opt, titles_opt = [], []
            for (m, z_index_opt) in zip(info['m_opt_list'], info['i_opt_list']):
                z_opt = self.ind_to_poi(z_index_opt)
                x_opt = self.gen.run(z_opt)
                y_opt = self.model.run_target(x_opt)
                title_opt = f'y = {y_opt:-10.4e} | m = {m:-7.1e}'
                X_opt.append(x_opt)
                titles_opt.append(title_opt)

            self.data.animate(X_opt, titles_opt,
                fpath=self.get_path(f'gif/out_max_protes_class{cl}.gif'))

        self.data.plot_many(X, titles, cols=5, rows=2,
            fpath=self.get_path('img/out_max_protes_all.png'))

    def run_data(self):
        for name_data in ['mnist', 'mnistf', 'cifar10', 'imagenet']:
            tm = self.log.prc(f'Check "{name_data}" dataset')
            data = self.data[name_data]
            v = len(data.labels)
            self.log(f'Number of classes   : {v:-10d}')
            if name_data != 'imagenet':
                v = len(data.data_trn)
                self.log(f'Size of trn dataset : {v:-10d}')
                v = len(data.data_tst)
                self.log(f'Size of tst dataset : {v:-10d}')
                data.plot_many(fpath=self.get_path(f'img/{name_data}.png'))
            self.log.res(tpc()-tm)

    def set_device(self, device=None):
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device

    def set_log(self):
        self.log = Log(self.get_path(f'log_{self.name}.txt'))
        self.log.title(f'Start "{self.name}" task (device "{self.device}").')

    def set_path(self, root_result='result'):
        self.path_root = os.path.dirname(__file__)
        self.path_result = os.path.join(self.path_root, root_result)
        os.makedirs(self.path_result, exist_ok=True)

        self.path_result = os.path.join(self.path_result, self.name)
        os.makedirs(self.path_result, exist_ok=True)
        os.makedirs(os.path.join(self.path_result, 'gif'), exist_ok=True)
        os.makedirs(os.path.join(self.path_result, 'img'), exist_ok=True)
        # os.makedirs(os.path.join(self.path_result, 'log'), exist_ok=True)
        # os.makedirs(os.path.join(self.path_result, 'dat'), exist_ok=True)

    def set_rand(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    def tmp1(self):
        #model = Model('vgg16', data)
        data = Data('imagenet')
        x = data.img_load('demo_image.jpg')
        print(x.shape)
        data.plot(x, 'Transformed', fpath=f'tmp.png')

    def tmp2(self):
        data = Data('cifar10')
        x, c = data.get(42)
        data.plot(x, fpath='tmp.jpg')
        x = data.img_load('demo_image.jpg')
        data.plot(x, 'Transformed', fpath=f'tmp.jpg')

    def tmp3(self):
        samples = 25
        z = torch.randn(samples, gen.d).to(device)
        x = gen.run(z)
        y = model.run(x)
        p = torch.argmax(y, axis=1).detach().to('cpu').numpy()
        l = [data.labels[p_cur] for p_cur in p]
        print(l)

        data.plot_many(x, l, cols=5, rows=5, fpath=f'result_tmp/gen_random.png')

    def tmp4(self, trn=False, tst=True):
        for mod in ['trn', 'tst']:
            if mod == 'trn' and not trn or mod == 'tst' and not tst:
                continue

            t = tpc()
            n, m = self.model.check(tst=mod == 'tst')
            t = tpc() - t

            text = f'Accuracy {mod}'
            text += f' : {float(m)/n*100:.2f}% ({m:-8d} / {n:-8d})'
            text += f' | time = {t:-10.2f} sec'
            print(text)


if __name__ == '__main__':
    name = sys.argv[1] if len(sys.argv) > 1 else NAMES[0]
    man = Manager(name)
