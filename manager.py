import numpy as np
import os
import random
import sys
from time import perf_counter as tpc
import torch


from data import Data
from gen import Gen
from model import Model
from opt import opt_ng_opo
from opt import opt_ng_portfolio
from opt import opt_ng_pso
from opt import opt_ng_spsa
from opt import opt_protes
from opt import opt_ttopt
from utils import Log


NAMES = [
    'data_check',
    'densenet_check', 'densenet_out',
    'gan_sn_check', 'gan_sn_inv']


OPTS = {
    'NG-OPO': opt_ng_opo,
    'NG-Portfolio': opt_ng_portfolio,
    'NG-PSO': opt_ng_pso,
    'NG-SPSA': opt_ng_spsa,
    'PROTES': opt_protes,
    'TTOpt': opt_ttopt,
}


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
        self.end()

    def end(self):
        self.log.end()

    def func(self, z, with_score=False):
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
        if self.name in ['data_check']:
            self.data = {}
            for name_data in ['mnist', 'mnistf', 'cifar10', 'imagenet']:
                tm = self.log.prc(f'Loading "{name_data}" dataset')
                self.data[name_data] = Data(name_data)
                self.log.res(tpc()-tm)

        if self.name in ['densenet_check']:
            tm = self.log.prc(f'Loading "cifar10" dataset')
            self.data = Data('cifar10')
            self.log.res(tpc()-tm)

            tm = self.log.prc(f'Loading "densenet" model')
            self.model = Model('densenet', self.data, self.device)
            self.log.res(tpc()-tm)

        if self.name in ['densenet_out']:
            tm = self.log.prc(f'Loading "cifar10" dataset')
            self.data = Data('cifar10')
            self.log.res(tpc()-tm)

            tm = self.log.prc(f'Loading "densenet" model')
            self.model = Model('densenet', self.data, self.device)
            self.log.res(tpc()-tm)

            tm = self.log.prc(f'Loading "gan_sn" generator')
            self.gen = Gen('gan_sn', self.device)
            self.log.res(tpc()-tm)

        if self.name in ['gan_sn_check']:
            tm = self.log.prc(f'Loading "cifar10" dataset')
            self.data = Data('cifar10')
            self.log.res(tpc()-tm)

            tm = self.log.prc(f'Loading "densenet" model')
            self.model = Model('densenet', self.data, self.device)
            self.log.res(tpc()-tm)

            tm = self.log.prc(f'Loading "gan_sn" generator')
            self.gen = Gen('gan_sn', self.device)
            self.log.res(tpc()-tm)

        if self.name in ['gan_sn_inv']:
            tm = self.log.prc(f'Loading "cifar10" dataset')
            self.data = Data('cifar10')
            self.log.res(tpc()-tm)

            tm = self.log.prc(f'Loading "gan_sn" generator')
            self.gen = Gen('gan_sn', self.device)
            self.log.res(tpc()-tm)

        self.log('')

    def run_data_check(self):
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

    def run_densenet_check(self, trn=True, tst=True):
        for mod in ['trn', 'tst']:
            if mod == 'trn' and not trn or mod == 'tst' and not tst:
                continue

            t = tpc()
            n, m = self.model.check(tst=(mod == 'tst'))
            t = tpc() - t

            text = f'Accuracy {mod}'
            text += f' : {float(m)/n*100:.2f}% ({m:-9d} / {n:-9d})'
            text += f' | time = {t:-10.2f} sec'
            self.log(text)

    def run_densenet_out(self, m=1.E+4):
        for meth, opt in OPTS.items():
            tm = self.log.prc(f'Start optimization with "{meth}"')

            X, titles = [], []
            for cl in np.arange(10):
                self.log.prc(f'Optimize class {cl} ({self.data.labels[cl]}) : ')
                self.model.set_target(cl=cl)

                t = tpc()
                z_index, e, hist = opt(self.func_ind, self.gen.d, self.n, m,
                    is_max=True)
                t = tpc() - t
                z = self.ind_to_poi(z_index)
                x = self.gen.run(z)
                y = self.model.run_target(x)

                self.log(f'Prob  : {e:-8.1e}')
                self.log(f'Iters : {m:-8.1e}')
                self.log(f'Time  : {t:-8.1e}')

                title = f'{self.data.labels[cl]} ({y:-9.3e})'
                X.append(x)
                titles.append(title)

                X_opt, titles_opt = [], []
                for (m_opt, z_index_opt, e_opt) in zip(*hist):
                    z_opt = self.ind_to_poi(z_index_opt)
                    x_opt = self.gen.run(z_opt)
                    title_opt = f'GAN (p={e_opt:-9.3e}; m={m_opt:-7.1e})'
                    X_opt.append(x_opt)
                    titles_opt.append(title_opt)

                fname = f'gif/gan_max_cl{cl}_{meth}.gif'
                self.data.animate(X_opt, titles_opt, fpath=self.get_path(fname))

            self.data.plot_many(X, titles, cols=5, rows=2,
                fpath=self.get_path(f'img/gan_max_cl{cl}_{meth}.png'))

            self.log.res(tpc()-tm)

    def run_gan_sn_check(self):
        for i in range(5):
            z = torch.randn(16, self.gen.d)
            x = self.gen.run(z)
            y = self.model.run(x).detach().to('cpu').numpy()
            c = np.argmax(y, axis=1)
            p = [y[i, c_cur] for i, c_cur in enumerate(c)]
            l = [self.data.labels[c_cur] for c_cur in c]
            titles = [f'{v_l} ({v_p:-7.1e})' for (v_p, v_l) in zip(p, l)]
            self.data.plot_many(x, titles, cols=4, rows=4,
                fpath=self.get_path(f'img/gan_sample_rand_{i+1}.png'))

    def run_gan_sn_inv(self, m=1.E+6, i_list=[1, 42, 99, 100, 700]):
        loss_img = torch.nn.MSELoss()

        for i in i_list:
            tm_out = self.log.prc(f'Start for trn image #{i:-6d}')
            x_real, c_real, l_real = self.data.get(i)

            def func(z):
                x = self.gen.run(z)
                e = [loss_img(x_cur, x_real) for x_cur in x]
                return np.array(e)

            def func_ind(z_index):
                return func(self.ind_to_poi(z_index))

            for meth, opt in OPTS.items():
                tm = self.log.prc(f'Start optimization with "{meth}"')

                t = tpc()
                z_index, e, hist = opt(func_ind, self.gen.d, self.n, m,
                    is_max=False)
                t = tpc() - t
                z = self.ind_to_poi(z_index)
                x = self.gen.run(z)

                self.log(f'Error : {e:-8.1e}')
                self.log(f'Iters : {m:-8.1e}')
                self.log(f'Time  : {t:-8.1e}')

                fname = f'img/gan_inv_img{i}_{meth}.png'
                self.data.plot_many(
                    [x_real, x],
                    [f'Target ({l_real})', f'GAN inv. (e={e:-9.3e})'],
                    cols=2, rows=1, fpath=self.get_path(fname))

                X_opt, titles_opt = [], []
                for (m_opt, z_index_opt, e_opt) in zip(*hist):
                    z_opt = self.ind_to_poi(z_index_opt)
                    x_opt = self.gen.run(z_opt)
                    title_opt = f'GAN (e={e_opt:-9.3e}; m={m_opt:-7.1e})'
                    X_opt.append(x_opt)
                    titles_opt.append(title_opt)

                fname = f'gif/gan_inv_img{i}_{meth}.gif'
                self.data.animate(X_opt, titles_opt, fpath=self.get_path(fname))

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
        self.log.title(f'Start "{self.name}" ({self.device}).')

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


if __name__ == '__main__':
    names = sys.argv[1:] if len(sys.argv) > 1 else NAMES
    for name in names:
        man = Manager(name)
