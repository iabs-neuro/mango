import numpy as np
import os
import random
import sys
import teneva
from time import perf_counter as tpc
import torch


# For faster and more accurate PROTES optimizer:
from jax.config import config
config.update('jax_enable_x64', True)
os.environ['JAX_PLATFORM_NAME'] = 'cpu'


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
from utils import plot_hist_am


AM_TARGET = {'layer': None, 'filter': None, 'cl': 0}
NAME_DATA = 'cifar10'
NAME_MODEL = 'densenet'
NAME_GEN = 'vae_vq' #  'gan_sn'
TASKS = [
    'check_data', 'check_gen', 'check_model',
]


OPTS = {
    'NG-OPO': opt_ng_opo,
    'NG-Portfolio': opt_ng_portfolio,
    'NG-PSO': opt_ng_pso,
    'NG-SPSA': opt_ng_spsa,
    'PROTES': opt_protes,
    'TTOpt': opt_ttopt,
}


class Manager:
    def __init__(self, name, device=None):
        if not name in TASKS:
            raise ValueError(f'Manager name "{name}" is not supported')

        self.name = name

        self.set_rand()
        self.set_device(device)
        self.set_path()
        self.set_log()

        self.load()

        eval('self.run_' + self.name + '()')

        self.end()

    def end(self):
        self.log.end()

    def func(self, z):
        x = self.gen.run(z)
        a = self.model.run_target(x).detach().to('cpu').numpy()
        return a

    def func_ind(self, z_index):
        return self.func(self.gen.ind_to_poi(z_index))

    def get_path(self, fpath):
        fpath = os.path.join(self.path_result, fpath)
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        return fpath

    def load(self, data='cifar10', model='densenet'):
        tm = self.log.prc(f'Loading "{NAME_DATA}" dataset')
        self.data = Data(NAME_DATA) #, m=(0.5,0.5,0.5), v=(1.0,1.0,1.0))
        self.log.res(tpc()-tm)

        tm = self.log.prc(f'Loading "{NAME_MODEL}" model')
        self.model = Model(NAME_MODEL, self.data, self.device)
        self.model.set_target(**AM_TARGET)
        self.log.res(tpc()-tm)

        tm = self.log.prc(f'Loading "{NAME_GEN}" generator')
        self.gen = Gen(NAME_GEN, self.data, self.device)
        self.log.res(tpc()-tm)

        self.log('')

    def run_check_data(self):
        name = self.data.name

        tm = self.log.prc(f'Check data for "{name}" dataset')

        v = len(self.data.labels)
        self.log(f'Number of classes   : {v:-10d}')

        if name != 'imagenet':
            v = len(self.data.data_trn)
            self.log(f'Size of trn dataset : {v:-10d}')

            v = len(self.data.data_tst)
            self.log(f'Size of tst dataset : {v:-10d}')

            self.data.plot_many(fpath=self.get_path(f'img/{name}.png'))

        self.log.res(tpc()-tm)

    def run_check_gen(self, m1=5, m2=5, rep=5):
        for i in range(rep):
            if self.gen.discrete:
                z = teneva.sample_lhs([self.gen.n]*self.gen.d, m1*m2)
            else:
                z = torch.randn(m1*m2, self.gen.d)

            t = tpc()
            x = self.gen.run(z)
            t = (tpc() - t) / len(x)

            self.log(f'Gen {len(x)} random samples (time/sample {t:-8.5f} sec)')

            p, l = self.model.run_pred(x)
            titles = [f'{v_l} ({v_p:-7.1e})' for (v_p, v_l) in zip(p, l)]

            self.data.plot_many(x, titles, cols=m1, rows=m2,
                fpath=self.get_path(f'img/{i+1}/gen_rand.png'))

        if self.gen.enc is None:
            return

        self.log('')

        for i in range(rep):
            x = torch.cat([self.data.get()[0][None] for _ in range(m1*m2)])
            p, l = self.model.run_pred(x)
            titles = [f'{v_l} ({v_p:-7.1e})' for (v_p, v_l) in zip(p, l)]

            self.data.plot_many(x, titles, cols=m1, rows=m2,
                fpath=self.get_path(f'img/{i+1}/gen_real.png'))

            t = tpc()
            z = self.gen.enc(x)
            t = (tpc() - t) / len(x)

            self.log(f'Gen {len(x)} embeddings     (time/sample {t:-8.5f} sec)')

            x = self.gen.run(z)
            p, l = self.model.run_pred(x)
            titles = [f'{v_l} ({v_p:-7.1e})' for (v_p, v_l) in zip(p, l)]

            self.data.plot_many(x, titles, cols=m1, rows=m2,
                fpath=self.get_path(f'img/{i+1}/gen_repr.png'))

    def run_check_model(self, trn=True, tst=True):
        for mod in ['trn', 'tst']:
            if mod == 'trn' and not trn or mod == 'tst' and not tst:
                continue

            t = tpc()
            n, m, a = self.model.check(tst=(mod == 'tst'),
                only_one_batch=(str(self.device)=='cpu'), with_target=True)
            t = tpc() - t

            text = f'Accuracy   {mod}'
            text += f' : {float(m)/n*100:.2f}% ({m:-9d} / {n:-9d})'
            text += f' | time = {t:-10.2f} sec'
            self.log(text)

            text = f'Activation {mod}'
            text += f' : [{np.min(a):-7.1e}, {np.max(a):-7.1e}] '
            text += f'(avg: {np.mean(a):-7.1e})'
            self.log(text)

            title = f'Activation of the target neuron on the "{mod}" data'
            fpath = self.get_path(f'img/check_target_{mod}.png')
            plot_hist_am(a, title, fpath)

            self.log()

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
        self.path_result = os.path.join(self.path_root, root_result, self.name)

    def set_rand(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)


if __name__ == '__main__':
    names = sys.argv[1:] if len(sys.argv) > 1 else TASKS
    for name in names:
        man = Manager(name)
