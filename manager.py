import argparse
import numpy as np
import os
import pickle
import random
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
from utils import plot_opt_conv


OPTS = {
    'Portfolio': {
        'func': opt_ng_portfolio,
    },
    'PROTES': {
        'func': opt_protes,
    },
    'TTOpt': {
        'func': opt_ttopt,
    },
    'Q-PROTES-small': {
        'func': opt_protes,
        'args': {'k': 10, 'k_top': 1, 'with_qtt': True},
    },
    'Q-PROTES': {
        'func': opt_protes,
        'args': {'with_qtt': True},
    },
    'Q-TTOpt': {
        'func': opt_ttopt,
        'args': {'with_qtt': True},
    },
    'Q-PROTES-small-2': {
        'func': opt_protes,
        'args': {'k': 10, 'k_top': 2, 'with_qtt': True},
    },
    'PROTES-small': {
        'func': opt_protes,
        'args': {'k': 10, 'k_top': 1},
    },
    'PROTES-small-2': {
        'func': opt_protes,
        'args': {'k': 10, 'k_top': 2},
    },
}


OPTS = {
    'Portfolio': {
        'func': opt_ng_portfolio,
    },
    'PROTES': {
        'func': opt_protes,
    },
    'TTOpt': {
        'func': opt_ttopt,
    },
    'Q-PROTES-small': {
        'func': opt_protes,
        'args': {'k': 10, 'k_top': 1, 'with_qtt': True},
    },
    'Q-PROTES': {
        'func': opt_protes,
        'args': {'with_qtt': True},
    },
    'Q-TTOpt': {
        'func': opt_ttopt,
        'args': {'with_qtt': True},
    },
    'Q-PROTES-small-2': {
        'func': opt_protes,
        'args': {'k': 10, 'k_top': 2, 'with_qtt': True},
    },
    'PROTES-small': {
        'func': opt_protes,
        'args': {'k': 10, 'k_top': 1},
    },
    'PROTES-small-2': {
        'func': opt_protes,
        'args': {'k': 10, 'k_top': 2},
    },
}


class Manager:
    def __init__(self, data, gen, model, task, kind, c=None, l=None, f=None, root='result', device=None):
        self.data_name = data
        self.gen_name = gen
        self.model_name = model
        self.task = task
        self.kind = kind
        self.c = c
        self.l = l
        self.f = f

        self.set_rand()
        self.set_device(device)
        self.set_path(root)
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
        fpath = os.path.join(self.path, fpath)
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        return fpath

    def load_data(self, log=True):
        if self.data_name is None:
            raise ValueError('Name of the dataset is not set')
        if log:
            tm = self.log.prc(f'Loading "{self.data_name}" dataset')
        try:
            self.data = Data(self.data_name)
            if log:
                self.log.res(tpc()-tm)
        except Exception as e:
            self.log.wrn('Can not load Data')
        if log:
            self.log('')

    def load_gen(self, log=True):
        if self.gen_name is None:
            return
        if log:
            tm = self.log.prc(f'Loading "{self.gen_name}" generator')
        try:
            self.gen = Gen(self.gen_name, self.data, self.device)
            if log:
                self.log.res(tpc()-tm)
        except Exception as e:
            self.log.wrn('Can not load Gen')
        if log:
            self.log('')

    def load_model(self, log=True):
        if self.model_name is None:
            return
        if log:
            tm = self.log.prc(f'Loading "{self.model_name}" model')
        try:
            self.model = Model(self.model_name, self.data, self.device)
            self.model.set_target(self.c, self.l, self.f)
            if log:
                self.log.res(tpc()-tm)
        except Exception as e:
            self.log.wrn(f'Can not load Model')
        if log:
            self.log('')

    def run_train_cifar10_vae_vq(self, lr=1.E-3, iters=15000, log_step=500):
        from gen.vae_vq_cifar10 import VAEVqCifar10
        tm = self.log.prc(f'Training "vae_vq_cifar10" model')

        vae = VAEVqCifar10()
        vae.to(self.device)

        optimizer = optim.Adam(vae.parameters(), lr=lr, amsgrad=False)

        train_res_recon_error = []
        train_res_perplexity = []

        vae.train()

        # Batch of real images to visualize accuracy while training:
        x_real = torch.cat([self.data.get()[0][None] for _ in range(25)])
        p, l = self.model.run_pred(x_real)
        titles = [f'{v_l} ({v_p:-7.1e})' for (v_p, v_l) in zip(p, l)]
        self.data.plot_many(x_real, titles, cols=5, rows=5,
            fpath=self.get_path(f'img/images_real.png'))

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
        info = ''
        if self.data_name:
            info += f'Data                : "{self.data_name}"\n'
        if self.gen_name:
            info += f'Gen                 : "{self.gen_name}"\n'
        if self.model_name:
            info += f'Model               : "{self.model_name}"\n'
        if self.task:
            info += f'Task                : "{self.task}"\n'
        if self.kind:
            info += f'Kind of task        : "{self.kind}"\n'
        if self.c:
            info += f'Target class        : "{self.c}"\n'
        if self.l:
            info += f'Target layer        : "{self.l}"\n'
        if self.f:
            info += f'Target filter       : "{self.f}"\n'

        self.log = Log(self.get_path(f'log.txt'))
        self.log.title(f'Computations ({self.device})', info)

    def set_path(self, root='result'):
        fbase = f'{self.data_name}'
        if self.gen_name:
            fbase += f'-{self.gen_name}'
        if self.model_name:
            fbase += f'-{self.model_name}'

        ftask = f'{self.task}-{self.kind}'
        if self.c:
            ftask += f'-c_{self.c}'
        if self.l:
            ftask += f'-l_{self.l}'
        if self.f:
            ftask += f'-f_{self.f}'

        self.path = os.path.join(root, fbase, ftask)

    def set_rand(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    def task_am_class(self, m=1.E+4, m_short=1.E+3):
        c = int(self.c)
        l = self.data.labels[c]
        tm = self.log.prc(f'Run AM for out class "{c}" ({l})')

        X, titles, res = [], [], {}
        for meth, opt in OPTS.items():
            self.log(f'\nOptimization with "{meth}" method:')
            self.model.set_target(c=c)

            t = tpc()
            func = opt.get('func')
            args = opt.get('args', {})
            z_index, _, hist = func(self.func_ind, self.gen.d, self.gen.n, m,
                is_max=True, **args)
            res[meth] = hist
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
            fname = f'gif/am_c{c}_{meth}.gif'
            self.data.animate(X_opt, titles_opt, fpath=self.get_path(fname))

        with open(self.get_path('dat/opt_info.pkl'), 'wb') as f:
            pickle.dump(res, f)

        # with open(self.get_path('dat/opt_info.pkl'), 'rb') as f:
        #     res = pickle.load(f)

        title = f'Activation maximization for class "{c}" ({l})'
        plot_opt_conv(res, title, self.get_path('img/opt_conv.png'))
        try:
            plot_opt_conv(res, title, self.get_path('img/opt_conv_short.png'),
                m_min=m_short)
        except Exception as e:
            pass

        fname = f'img/am_c{c}.png'
        self.data.plot_many(X, titles, fpath=self.get_path(fname),
            cols=len(X), rows=1, size=4)

        self.log.res(tpc()-tm)

    def task_check_data(self):
        name = self.data.name
        tm = self.log.prc(f'Check data for "{name}" dataset')
        self.log(self.data.info())
        if name != 'imagenet':
            self.data.plot_many(fpath=self.get_path(f'img/{name}.png'))
        self.log.res(tpc()-tm)

    def task_check_gen(self, m1=5, m2=5, rep=5):
        tm = self.log.prc(f'Generate random images')

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
                fpath=self.get_path(f'img/{i+1}_gen_rand.png'))

        self.log.res(tpc()-tm)

        if self.gen.enc is None:
            return

        self.log('')

        tm = self.log.prc(f'Reconstruct images from the dataset')

        for i in range(rep):
            x = torch.cat([self.data.get()[0][None] for _ in range(m1*m2)])
            p, l = self.model.run_pred(x)
            titles = [f'{v_l} ({v_p:-7.1e})' for (v_p, v_l) in zip(p, l)]

            self.data.plot_many(x, titles, cols=m1, rows=m2,
                fpath=self.get_path(f'img/{i+1}_gen_real.png'))

            t = tpc()
            z = self.gen.rev(x)
            t = (tpc() - t) / len(x)

            self.log(f'Gen {len(x)} embeddings     (time/sample {t:-8.2e} sec)')

            x = self.gen.run(z)
            p, l = self.model.run_pred(x)
            titles = [f'{v_l} ({v_p:-7.1e})' for (v_p, v_l) in zip(p, l)]

            self.data.plot_many(x, titles, cols=m1, rows=m2,
                fpath=self.get_path(f'img/{i+1}_gen_repr.png'))

        self.log.res(tpc()-tm)

    def task_check_model(self, trn=True, tst=True):
        if not self.model.has_target():
            raise ValueError('Target for the model is not set')

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

            self.log('')

    def task_train_gen(self):
        if self.gen_name == 'vae_vq':
            if self.data_name == 'cifar10':
                return self.run_train_cifar10_vae_vq()

        raise NotImplementedError()


def args_build():
    parser = argparse.ArgumentParser(
        prog='neural_tensor_train',
        description='Software product for analysis of activations and specialization in artificial neural networks, including spiking neural networks, with the tensor train decomposition.',
        epilog = '© Andrei Chertkov'
    )
    parser.add_argument('-d', '--data',
        type=str,
        help='Name of the used dataset',
        default='cifar10',
        choices=['mnist', 'mnistf', 'cifar10', 'imagenet']
    )
    parser.add_argument('-g', '--gen',
        type=str,
        help='Name of the used generator',
        default=None,
        choices=['gan_sn', 'vae_vq']
    )
    parser.add_argument('-m', '--model',
        type=str,
        help='Name of the used model',
        default=None,
        choices=['densenet', 'vgg16']
    )
    parser.add_argument('-t', '--task',
        type=str,
        help='Name of the task',
        default=None,
        choices=['check', 'train', 'am']
    )
    parser.add_argument('-k', '--kind',
        type=str,
        help='Kind of the task',
        default=None,
    )
    parser.add_argument('-c', '--c',
        type=str,
        help='Target class',
        default=None,
    )
    parser.add_argument('-l', '--l',
        type=str,
        help='Target layer',
        default=None,
    )
    parser.add_argument('-f', '--f',
        type=str,
        help='Target filter',
        default=None,
    )
    parser.add_argument('-r', '--root',
        type=str,
        help='Path to the folder with results',
        default='result'
    )

    args = parser.parse_args()
    return args.data, args.gen, args.model, args.task, args.kind, args.c, args.l, args.f, args.root


if __name__ == '__main__':
    Manager(*args_build())
