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
from spikingjelly.activation_based.functional import reset_net


from .data.data_main import Data
from .gen.gen_main import Gen
from .model.model_main import Model
from .opt import opt_ng_portfolio, opt_protes
from .utils import Log, plot_hist_am, plot_opt_conv
from .hardware import configure_hardware
import os

configure_hardware(verbose=False)

OPTS = {
    'NG': {
        'func': opt_ng_portfolio,
    },
    'TT': {
        'func': opt_protes,
        'args': {'k': 10, 'k_top': 1, 'with_qtt': True},
    },
    'TT-s': {
        'func': opt_protes,
        'args': {'k': 5, 'k_top': 1, 'with_qtt': True}
    },
    'TT-b': {
        'func': opt_protes,
        'args': {'k': 25, 'k_top': 5, 'with_qtt': True}
    },
    'TT-exp': {
        'func': opt_protes,
        'args': {'k': 100, 'k_top': 10, 'with_qtt': True, 'lr': 0.1, 'k_gd': 1}
    },
}

PLOT_SHAPE = {
    1: (1,1),
    2: (1,2),
    3: (1,3),
    4: (2,2)
}


class MangoManager:
    def __init__(self, data, gen, model, task, kind, cls=None, layer=None,
                 unit=None, root='result', device=None, opt_args=None, model_path=None):

        self.data_name = data
        self.gen_name = gen
        self.model_name = model
        self.task = task
        self.kind = kind
        self.cls = cls
        self.layer = layer
        self.unit = unit
        self.opt_args = opt_args
        if 'am_methods' not in opt_args:
            self.opt_args['am_methods'] = list(OPTS.keys())
        if 'opt_budget' not in opt_args:
            self.opt_args['opt_budget'] = 10000
        if 'track_opt_progress' not in opt_args:
            self.opt_args['track_opt_progress'] = True
        if 'res_mode' not in opt_args:
            self.opt_args['res_mode'] = 'best'
        if 'nrep' not in opt_args:
            self.opt_args['nrep'] = 1

        print(self.opt_args)

        self.set_rand()
        self.set_device(device)
        self.set_path(root)
        self.set_log()
        self.load_data()
        self.load_gen()
        self.load_model(model_path=model_path)

    def end(self):
        self.log.end()

    def func(self, z):
        from spikingjelly.activation_based.functional import reset_net
        nrep = self.opt_args['nrep']
        x = self.gen.run(z)

        all_values = []
        for i in range(nrep):
            activation = self.model.run_target(x)
            #activation = self.model.run(x)[:,5]
            #reset_net(self.model.net)
            values = activation.detach().to('cpu').numpy()
            all_values.append(values)

        #print('values:', all_values)
        agg_values = np.vstack(all_values)
        #print(agg_values.shape)
        #print(np.mean(agg_values, axis=0))
        return np.mean(agg_values, axis=0)

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
        self.data = Data(self.data_name)
        if log:
            self.log('')

    def load_gen(self, log=True):
        if self.gen_name is None:
            return

        try:
            self.gen = Gen(self.gen_name, self.data, self.device)
            if log:
                tm = self.log.prc(f'Loading "{self.gen_name}" generator')
                self.log.res(tpc()-tm)

        except Exception as e:
            self.log(repr(e))
            self.log.wrn('Can not load Gen')

        if log:
            self.log('')

    def load_model(self, log=True, model_path=None):
        if self.model_name is None:
            return

        try:
            self.model = Model(self.model_name, self.data, self.device, model_path=model_path)
            self.model.set_target(cls=self.cls, layer=self.layer, unit=self.unit, logger=self.log)
            if log:
                tm = self.log.prc(f'Loading "{self.model_name}" model')
                self.log.res(tpc()-tm)

        except Exception as e:
            self.log(repr(e))
            self.log.wrn(f'Can not load Model')

        if log:
            self.log('')

    def run(self):
        method_name = f'task_{self.task}_{self.kind}'
        getattr(self, method_name)()
        self.end()

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
        p, _, l = self.model.run_pred(x_real)
        titles = [f'{v_l} ({v_p:-7.1e})' for (v_p, v_l) in zip(p, l)]
        self.data.plot_many(x_real,
                            titles,
                            cols=5,
                            rows=5,
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
                p, _, l = self.model.run_pred(x)
                titles = [f'{v_l} ({v_p:-7.1e})' for (v_p, v_l) in zip(p, l)]
                self.data.plot_many(x,
                                    titles,
                                    cols=5,
                                    rows=5,
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
        if self.cls:
            info += f'Target class        : "{self.cls}"\n'
        if self.layer:
            info += f'Target layer        : "{self.layer}"\n'
        if self.unit:
            info += f'Target unit       : "{self.unit}"\n'

        self.log = Log(self.get_path(f'log.txt'))
        self.log.title(f'Computations ({self.device})', info)

    def set_path(self, root='result'):
        fbase = f'{self.data_name}'
        if self.gen_name:
            fbase += f'-{self.gen_name}'
        if self.model_name:
            fbase += f'-{self.model_name}'

        ftask = f'{self.task}-{self.kind}'
        if self.cls is not None:
            ftask += f'-c_{self.cls}'
        if self.layer is not None:
            ftask += f'-l_{self.layer}'
        if self.unit is not None:
            ftask += f'-f_{self.unit}'

        self.path = os.path.join(root, fbase, ftask)

    def set_rand(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    def _task_am(self, m=1.E+4, m_short=1.E+3, mode='class', track_opt_progress=True):

        if self.opt_args['opt_budget'] is not None:
            m = self.opt_args['opt_budget']

        if mode == 'class':
            cls = int(self.cls)
            label = self.data.labels[cls]
            self.model.set_target(cls=cls, logger=self.log)
            tm = self.log.prc(f'Running AM for target class {cls} ({label})...')

        elif mode == 'unit':
            unit = int(self.unit)
            layer = self.layer
            self.model.set_target(layer=layer, unit=unit, logger=self.log)
            tm = self.log.prc(f'Running AM for unit {unit} of layer {self.layer} ({self.model.layer})...')

        X, titles, res = [], [], {}
        for i, meth in enumerate(self.opt_args['am_methods']):
            self.log(f'\nOptimization with "{meth}" method:')

            if meth in OPTS:
                opt = OPTS[meth]
                t = tpc()
                optimizer = opt.get('func')
                args = opt.get('args', {})

                # TODO: add seed fixation
                seed = np.random.randint(10**6)
                z_index, _, hist = optimizer(self.func_ind, self.gen.d, self.gen.n, m, seed=seed, is_max=True, **args)

                if len(set(self.opt_args['am_methods'])) == len(self.opt_args['am_methods']):
                    textmeth = meth
                else:
                    textmeth = f'{i}_{meth}'

                res[textmeth] = hist
                t = tpc() - t

                print(self.opt_args['res_mode'])
                #print(z_index)

                recomputed_activations = np.zeros(len(hist[1]) + 1)
                xlist = []
                for j, z_ind in enumerate(hist[1]):
                    z = self.gen.ind_to_poi(z_ind)
                    x = self.gen.run(z)
                    a = self.model.run_target(x)
                    recomputed_activations[j] = a
                    xlist.append(x)

                z = self.gen.ind_to_poi(z_index)
                x = self.gen.run(z)
                a = self.model.run_target(x)
                recomputed_activations[-1] = a
                xlist.append(x)

                print('recomputed history of activations:')
                print(recomputed_activations)

                if self.opt_args['res_mode'] == 'best': # replace latest image with the best one
                    best_ind = np.argmax(recomputed_activations)
                    x = xlist[best_ind]
                    a = recomputed_activations[best_ind]

                self.log(f'Result: it {m:-7.1e}, t {t:-7.1e}, a {a:-11.5e}')

                title = f'{meth} : p={a:-9.3e}'
                X.append(x)
                titles.append(title)

                if track_opt_progress:
                    X_opt, titles_opt = [], []
                    for (m_opt, z_index_opt, e_opt) in zip(*hist):
                        z_opt = self.gen.ind_to_poi(z_index_opt)
                        x_opt = self.gen.run(z_opt)
                        title_opt = f'{meth} : p={e_opt:-9.3e}; m={m_opt:-7.1e}'
                        X_opt.append(x_opt)
                        titles_opt.append(title_opt)

                    if mode == 'class':
                        fname = f'gif/am_c{cls}_{meth}.gif'
                    elif mode == 'unit':
                        fname = f'gif/am_u{unit}_layer_({layer})_{meth}.gif'

                    self.data.animate(X_opt, titles_opt, fpath=self.get_path(fname))

            else:
                self.log(f'Unknown method {meth}, optimization skipped')

        with open(self.get_path('dat/opt_info.pkl'), 'wb') as f:
            pickle.dump(res, f)

        # with open(self.get_path('dat/opt_info.pkl'), 'rb') as f:
        #     res = pickle.load(f)

        if mode == 'class':
            title = f'Activation maximization for class "{cls}" ({label})'
        elif mode == 'unit':
            title = f'Activation maximization for unit {unit} of layer {self.layer}'

        try:
            plot_opt_conv(res, title, self.get_path('img/opt_conv.png'))
            plot_opt_conv(res, title, self.get_path('img/opt_conv_short.png'), m_min=m_short)
        except Exception as e:
            self.log(repr(e))
            self.log.wrn(f'AM plotting failed')

        if mode == 'class':
            fname = f'img/am_c{cls}.png'
        elif mode == 'unit':
            fname = f'img/am_u{unit}_{layer}.png'

        self.data.plot_many(X,
                            titles,
                            fpath=self.get_path(fname),
                            rows=PLOT_SHAPE[len(X)][0],
                            cols=PLOT_SHAPE[len(X)][1],
                            size=10)

        self.log.res(tpc()-tm)

    def task_am_class(self, m=1.E+4, m_short=1.E+3):
        if self.model.target_mode is None:
            self.model.set_target_mode(self.cls, self.layer, self.unit)
        if self.model.target_mode != 'class':
            raise ValueError(f'Input (class={self.cls}, unit={self.unit}, layer={self.layer}) not compatible with class AM mode')

        self._task_am(m, m_short, mode='class', track_opt_progress=self.opt_args['track_opt_progress'])

    def task_am_unit(self, m=1.E+4, m_short=1.E+3):
        if self.model.target_mode is None:
            self.model.set_target_mode(self.cls, self.layer, self.unit)
        if self.model.target_mode != 'unit':
            raise ValueError(
                f'Input (class={self.cls}, unit={self.unit}, layer={self.layer}) not compatible with unit AM mode')

        self._task_am(m, m_short, mode='unit', track_opt_progress=self.opt_args['track_opt_progress'])

    def task_check_data(self):
        name = self.data.name
        tm = self.log.prc(f'Check data for "{name}" dataset')
        self.log(self.data.info())
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

            p, _, l = self.model.run_pred(x)
            titles = [f'{v_l} ({v_p:-7.1e})' for (v_p, v_l) in zip(p, l)]

            self.data.plot_many(x,
                                titles,
                                cols=m1,
                                rows=m2,
                                fpath=self.get_path(f'img/{i+1}_gen_rand.png'))

        self.log.res(tpc()-tm)

        if self.gen.enc is None:
            return

        self.log('')

        tm = self.log.prc(f'Reconstruct images from the dataset')

        for i in range(rep):
            x = torch.cat([self.data.get()[0][None] for _ in range(m1*m2)])
            p, _, l = self.model.run_pred(x)
            titles = [f'{v_l} ({v_p:-7.1e})' for (v_p, v_l) in zip(p, l)]

            self.data.plot_many(x,
                                titles,
                                cols=m1,
                                rows=m2,
                                fpath=self.get_path(f'img/{i+1}_gen_real.png'))

            t = tpc()
            z = self.gen.rev(x)
            t = (tpc() - t) / len(x)

            self.log(f'Gen {len(x)} embeddings     (time/sample {t:-8.2e} sec)')

            x = self.gen.run(z)
            p, _, l = self.model.run_pred(x)
            titles = [f'{v_l} ({v_p:-7.1e})' for (v_p, v_l) in zip(p, l)]

            self.data.plot_many(x,
                                titles,
                                cols=m1,
                                rows=m2,
                                fpath=self.get_path(f'img/{i+1}_gen_repr.png'))

        self.log.res(tpc()-tm)

    def task_check_model(self, trn=True, tst=True):
        if not self.model.has_target():
            raise ValueError('Target for the model is not set')

        for mod in ['trn', 'tst']:
            if mod == 'trn' and not trn or mod == 'tst' and not tst:
                continue
            if mod == 'trn' and self.data.data_trn is None:
                continue
            if mod == 'tst' and self.data.data_tst is None:
                continue

            t = tpc()
            n, m, a = self.model.check(tst=(mod == 'tst'),
                                       only_one_batch=(str(self.device) == 'cpu'),
                                       with_target=True)
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

        raise NotImplementedError(f'task_train_gen not implemented for gen {self.gen_name} and dataset {self.data_name} pair')


def args_build():
    parser = argparse.ArgumentParser(
        prog='MANGO',
        description='Software product for analysis of activations and specialization in '\
                    'artificial neural networks (ANN), including spiking neural networks (SNN) '\
                    'with the tensor train (TT) decomposition and other gradient-free methods.',
        epilog = '© Andrei Chertkov, Nikita Pospelov, Maxim Beketov'
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
        choices=['alexnet', 'densenet', 'vgg16', 'vgg19', 'snn']
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
        default=None
    )
    parser.add_argument('-c', '--cls',
        type=str,
        help='Target class',
        default=None
    )
    parser.add_argument('-l', '--layer',
        type=str,
        help='Target layer',
        default=0
    )
    parser.add_argument('-u', '--unit',
        type=str,
        help='Target unit',
        default=0
    )
    parser.add_argument('-r', '--root',
        type=str,
        help='Path to the folder with results',
        default='result'
    )

    args = parser.parse_args()
    return args.data, args.gen, args.model, args.task, args.kind, args.cls, args.layer, args.unit, args.root


if __name__ == '__main__':
    MangoManager(*args_build()).run()
