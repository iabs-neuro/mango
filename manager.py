import numpy as np
import os
import random
import sys
import teneva
from time import perf_counter as tpc
import torch
import torch.nn.functional as F
import torch.optim as optim


# For faster and more accurate PROTES optimizer:
from jax.config import config
config.update('jax_enable_x64', True)
os.environ['JAX_PLATFORM_NAME'] = 'cpu'


from data import Data
from gen import Gen
from model import Model
from opt import opt_ng_portfolio
from opt import opt_protes
from opt import opt_ttopt
from utils import Log
from utils import plot_hist_am


AM_TARGET = {'layer': None, 'filter': None, 'cl': 0} # TODO!
OPTS = {
    'Portfolio': opt_ng_portfolio,
    'PROTES': opt_protes,
    'TTOpt': opt_ttopt,
}
TASKS = [
    'cifar10-vae_vq-densenet-check-data',
    'cifar10-vae_vq-densenet-train-gen',
    'cifar10-vae_vq-densenet-check-gen',
    'cifar10-vae_vq-densenet-check-model',
    'cifar10-vae_vq-densenet-am-cl',
]


class Manager:
    def __init__(self, name, device=None):
        if not name in TASKS:
            raise ValueError(f'Manager task "{name}" is not supported')

        self.name = name
        self.name_data = name.split('-')[0]
        self.name_gen = name.split('-')[1]
        self.name_model = name.split('-')[2]
        self.task = name.split('-')[3]
        self.kind = name.split('-')[4]

        self.set_rand()
        self.set_device(device)
        self.set_path()
        self.set_log()

        self.load_data()
        self.load_gen()
        self.load_model()

        eval(f'self.task_{self.task}_{self.kind}()')

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

    def load_data(self, name=None, log=True):
        name = name or self.name_data
        if log:
            tm = self.log.prc(f'Loading "{name}" dataset')
        try:
            self.data = Data(name)
            if log:
                self.log.res(tpc()-tm)
        except Exception as e:
            self.log.wrn('Can not load Data')
        if log:
            self.log('')

    def load_gen(self, name=None, log=True):
        name = name or self.name_gen
        if log:
            tm = self.log.prc(f'Loading "{name}" generator')
        try:
            self.gen = Gen(name, self.data, self.device)
            if log:
                self.log.res(tpc()-tm)
        except Exception as e:
            self.log.wrn('Can not load Gen')
        if log:
            self.log('')

    def load_model(self, name=None, log=True):
        name = name or self.name_model
        if log:
            tm = self.log.prc(f'Loading "{name}" model')
        try:
            self.model = Model(name, self.data, self.device)
            self.model.set_target(**AM_TARGET)
            if log:
                self.log.res(tpc()-tm)
        except Exception as e:
            self.log.wrn('Can not load Model')
        if log:
            self.log('')

    def run_train_vae_vq_cifar10(self, lr=1.E-3, iters=15000, log_step=500):
        from gen.vae_vq_cifar10 import VAEVqCifar10
        tm = self.log.prc(f'Training "vae_vq_cifar10" model')

        vae = VAEVqCifar10()
        vae.to(self.device)

        optimizer = optim.Adam(vae.parameters(), lr=lr, amsgrad=False)

        data_variance = np.var(self.data.data_trn.data / 255.0)

        train_res_recon_error = []
        train_res_perplexity = []

        vae.train()

        # Batch of real images to visualize accuracy while training:
        x_real = torch.cat([self.data.get()[0][None] for _ in range(25)])
        p, l = self.model.run_pred(x_real)
        titles = [f'{v_l} ({v_p:-7.1e})' for (v_p, v_l) in zip(p, l)]
        self.data.plot_many(x_real, titles, cols=5, rows=5,
            fpath=self.get_path(f'img/imgs_real.png'))

        for it in range(iters):
            (data, _) = next(iter(self.data.dataloader_trn))
            data = data.to(self.device)
            optimizer.zero_grad()

            vq_loss, data_recon, perplexity = vae(data)
            recon_error = F.mse_loss(data_recon, data) / self.data.var_trn
            loss = recon_error + vq_loss
            loss.backward()

            optimizer.step()

            train_res_recon_error.append(recon_error.item())
            train_res_perplexity.append(perplexity.item())

            if (it+1) % log_step == 0 or it == 0 or it == iters-1:
                fpath = 'gen/vae_vq_cifar10/data/vae_vq_cifar10.pt'
                torch.save(vae.state_dict(), fpath)

                e_recon = np.mean(train_res_recon_error[-log_step:])
                e_perpl = np.mean(train_res_perplexity[-log_step:])

                text = f'# {it+1:-8d} | '
                text += f'time {tpc()-tm:-7.1e} sec | '
                text += f'E recon = {e_recon:-9.3e} | '
                text += f'Perplexity = {e_perpl:-9.3e} | '
                self.log(text)

                # Plot samples for the current model:
                self.load_gen(log=False)
                z = self.gen.rev(x_real)
                x = self.gen.run(z)
                p, l = self.model.run_pred(x)
                titles = [f'{v_l} ({v_p:-7.1e})' for (v_p, v_l) in zip(p, l)]
                self.data.plot_many(x, titles, cols=5, rows=5,
                    fpath=self.get_path(f'img/it_{it+1}_gen.png'))

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
        self.log.title(f'{self.name} ({self.device})')

    def set_path(self, root_result='result'):
        self.path_root = os.path.dirname(__file__)
        self.path_result = os.path.join(self.path_root, root_result, self.name)

    def set_rand(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    def task_am_cl(self, m=1.E+4):
        c = AM_TARGET['cl']
        l = self.data.labels[c]
        tm = self.log.prc(f'Run AM for out class "{c}" ({l})')

        X, titles = [], []
        for meth, opt in OPTS.items():
            self.log(f'\nOptimization with "{meth}" method:')
            self.model.set_target(cl=c)

            t = tpc()
            z_index, _, hist = opt(self.func_ind, self.gen.d, self.gen.n, m,
                is_max=True)
            t = tpc() - t
            z = self.gen.ind_to_poi(z_index)
            x = self.gen.run(z)
            a = self.model.run_target(x)

            self.log(f'Result: it {m:-7.1e}, t {t:-7.1e}, a {a:-11.5e}')

            title = f'{meth} : p={a:-9.3e} ({l})'
            X.append(x)
            titles.append(title)

            X_opt, titles_opt = [], []
            for (m_opt, z_index_opt, e_opt) in zip(*hist):
                z_opt = self.gen.ind_to_poi(z_index_opt)
                x_opt = self.gen.run(z_opt)
                title_opt = f'{meth} : p={e_opt:-9.3e}; m={m_opt:-7.1e}'
                X_opt.append(x_opt)
                titles_opt.append(title_opt)
            fname = f'gif/am_cl{c}_{meth}.gif'
            self.data.animate(X_opt, titles_opt, fpath=self.get_path(fname))

        fname = f'img/am_cl{c}.png'
        self.data.plot_many(X, titles, fpath=self.get_path(fname),
            cols=len(X), rows=1)

        self.log.res(tpc()-tm)

    def task_check_data(self):
        name = self.data.name
        tm = self.log.prc(f'Check data for "{name}" dataset')
        self.log(self.data.info())
        if name != 'imagenet':
            self.data.plot_many(fpath=self.get_path(f'img/{name}.png'))
        self.log.res(tpc()-tm)

    def task_check_gen(self, m1=5, m2=5, rep=5):
        for i in range(rep):
            if self.gen.discrete:
                z = teneva.sample_lhs([self.gen.n]*self.gen.d, m1*m2)
            else:
                z = torch.randn(m1*m2, self.gen.d)

            t = tpc()
            x = self.gen.run(z)
            t = (tpc() - t) / len(x)

            self.log(f'Gen {len(x)} random samples (time/sample {t:-8.2e} sec)')

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
            z = self.gen.rev(x)
            t = (tpc() - t) / len(x)

            self.log(f'Gen {len(x)} embeddings     (time/sample {t:-8.2e} sec)')

            x = self.gen.run(z)
            p, l = self.model.run_pred(x)
            titles = [f'{v_l} ({v_p:-7.1e})' for (v_p, v_l) in zip(p, l)]

            self.data.plot_many(x, titles, cols=m1, rows=m2,
                fpath=self.get_path(f'img/{i+1}/gen_repr.png'))

    def task_check_model(self, trn=True, tst=True):
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

    def task_train_gen(self):
        if self.name_gen == 'vae_vq':
            if self.name_data == 'cifar10':
                return self.run_train_vae_vq_cifar10()

        raise NotImplementedError()


if __name__ == '__main__':
    tasks = sys.argv[1:] if len(sys.argv) > 1 else TASKS
    for task in tasks:
        man = Manager(task)
