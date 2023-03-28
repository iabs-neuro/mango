import numpy as np
import sys
from time import perf_counter as tpc


from .manager import Manager


sys.path.append('..')
from data import Data
from gen import Gen
from model import Model
from opt import opt_ng_portfolio
from opt import opt_protes


class ManagerDensenet(Manager):
    def __init__(self, name='densenet', device=None):
        super().__init__(name, device)

    def load(self):
        self.data = Data('cifar10')
        self.model = Model('densenet161', self.data, self.device)
        self.gen = Gen('gan-sn', self.device)


class ManagerDensenetCheck(ManagerDensenet):
    def __init__(self, name='densenet_check', device=None):
        super().__init__(name, device)

    def run(self, trn=False, tst=True):
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


class ManagerDensenetOut(ManagerDensenet):
    def __init__(self, name='densenet_out', device=None):
        super().__init__(name, device)

    def run(self, evals=1.E+4):
        X, titles = [], []
        for cl in np.arange(10):
            print(f'\n>>> Optimize class {cl} ({self.data.labels[cl]}) : ')
            self.model.set_target(cl=cl)
            z, info = opt_protes(self.model, self.gen, evals, with_score=True)
            x = self.gen.run(z)
            y = self.model.run_target(x)
            title = f'y={y:-8.2e} : {self.data.labels[cl]}'
            self.data.plot(x, title,
                fpath=self.get_path(f'out_max_protes_class{cl}.png'))
            X.append(x)
            titles.append(title)

        self.data.plot_many(X, titles, cols=5, rows=2,
            fpath=self.get_path('out_max_protes_all.png'))
