import numpy as np
import random
from time import perf_counter as tpc
import torch


from core import Data
from core import Model
from core.utils import Log
from core.utils import folder_ensure
from core.utils import sort_vector
from gen import GenGAN
from opt import opt_am
from opt import opt_protes
from opt import opt_ttopt


def demo(device, name='vgg16', layer=5, filter=20, evals=1.E+3):
    model, gen = run_init(device, name, layer, filter)

    x_base = model.img_load('demo_image.jpg')
    x_rand = model.img_rand()

    run_predict(model, x_base)
    run_predict(model, x_rand, is_rand=True)

    run_predict_gen(model, gen)

    run_opt_protes(model, gen, evals)
    run_opt_ttopt(model, gen, evals)
    run_opt_am(model, x_rand, evals)


def run_init(device, name, layer, filter, name_gen='fc8'):
    """Prepare data, model and generator."""
    _time = tpc()
    log.prc('Prepare data, model and generator')

    data = Data('imagenet')
    data.load_labels()

    model = Model(name, device, data.sz, data.ch, data.labels)
    model.set_target(layer, filter)
    log(f'Model is loaded ("{name}")')

    gen = GenGAN(name_gen, device, model.sz)
    log(f'GEN   is loaded ("{name_gen}"). Latent dimension: {gen.d:-3d}')

    log.res(tpc() - _time)

    return model, gen


def run_opt_protes(model, gen, evals):
    """Run optimization with PROTES."""
    _time = tpc()
    log.prc(f'Optimization with PROTES')

    z = opt_protes(model, gen, evals)
    x = gen.run(z)

    y = model.run(x)
    a = model.get_a()

    title = f'Activation {a:-9.2e} | Method PROTES'
    fpath = f'result/image/opt_protes.png'
    model.img_show(x, title, fpath)

    log(f'Activation in target neuron : {a:-14.8e}')
    log.res(tpc() - _time)


def run_opt_ttopt(model, gen, evals):
    """Run optimization with TTOpt."""
    _time = tpc()
    log.prc(f'Optimization with TTOpt')

    z = opt_ttopt(model, gen, evals)
    x = gen.run(z)

    y = model.run(x)
    a = model.get_a()

    title = f'Activation {a:-9.2e} | Method TTOpt'
    fpath = f'result/image/opt_ttopt.png'
    model.img_show(x, title, fpath)

    log(f'Activation in target neuron : {a:-14.8e}')
    log.res(tpc() - _time)


def run_opt_am(model, x, evals, comment=''):
    """Run optimization with Activation Maximization (AM)."""
    _time = tpc()
    log.prc(f'Optimization with Activation Maximization {comment}')

    x = opt_am(model, x, evals=int(evals))

    y = model.run(x)
    a = model.get_a()

    title = f'Activation {a:-9.2e} | Method AM'
    fpath = f'result/image/opt_am.png'
    model.img_show(x, title, fpath)

    log(f'Activation in target neuron : {a:-14.8e}')
    log.res(tpc() - _time)


def run_predict(model, x, is_rand=False):
    """Run model prediction."""
    _time = tpc()
    suf = ('Rand' if is_rand else 'Base') + ' image'
    log.prc(f'Simple ANN prediction for given input ({suf})')

    y = model.run(x)
    a = model.get_a()

    for lbl in sort_vector(y)[:3]:
        log(f'    {lbl[1]*100:-5.1f}% : {model.labels[lbl[0]]}')

    title = f'Activation {a:-9.2e} | {suf}'
    fpath = f'result/image/ann_pred_{"rand" if is_rand else "base"}.png'
    model.img_show(x, title, fpath)

    log(f'Activation in target neuron : {a:-14.8e}')
    log.res(tpc() - _time)


def run_predict_gen(model, gen):
    """Run model prediction with generator input."""
    _time = tpc()
    log.prc(f'Simple GEN->ANN prediction for random latent input')

    z = torch.zeros(gen.d, device=model.device)

    x = gen.run(z)

    y = model.run(x)
    a = model.get_a()

    for lbl in sort_vector(y)[:3]:
        log(f'    {lbl[1]*100:-5.1f}% : {model.labels[lbl[0]]}')

    title = f'Activation {a:-9.2e} | Random generated image'
    fpath = f'result/image/gen_pred_rand.png'
    model.img_show(x, title, fpath)

    log(f'Activation in target neuron : {a:-14.8e}')
    log.res(tpc() - _time)


if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    folder_ensure('result')
    folder_ensure('result/image')
    folder_ensure('result/log')

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    log = Log('result/log/demo.txt')
    log.title(f'Demo computations (device "{device}").')

    demo(device)
