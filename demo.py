import numpy as np
import random
from time import perf_counter as tpc
import torch


from core import Data
from core import Image
from core import Model
from core.utils import Log
from core.utils import folder_ensure
from core.utils import plot_image
from core.utils import resize_and_pad
from core.utils import sort_vector
from gen import GenGAN
from opt import opt_am
from opt import opt_protes
from opt import opt_ttopt


def demo(device, log, name='vgg16', layer=2, filter=10, evals=1.E+1):
    # Note, combination (name, layer, filter) is tested only for 'vgg16'.
    _time = tpc()

    data, model, gen, d = run_init(device, log, name, layer, filter)
    image_demo = Image('demo_image.jpg', 'file')
    image_rand = Image.rand(data.sz, data.ch, 50, 180)

    run_predict(data, model, image_demo, '(demo image)')
    run_predict(data, model, image_rand, '(random pixels)')

    #run_opt_ttopt(model, gen, d, evals)

    run_opt_am(model, image_rand, evals)


def run_init(device, log, name, layer, filter):
    """Prepare data, model and generator."""
    _time = tpc()
    log.prc('Prepare data, model and generator')

    data = Data('imagenet')
    data.load_labels()

    model = Model(device)
    model.set(name=name)
    model.set_labels(data.labels, data.name)
    model.set_shape(data.sz, data.ch)
    model.set_target(layer, filter)

    gen = GenGAN(name='fc8')
    d = gen.codelen

    log.res(tpc()-_time)

    return data, model, gen, d


def run_opt_ttopt(model, gen, d, evals, rank=4, comment=''):
    """Run optimization with TTOpt."""
    _time = tpc()
    log.prc(f'Optimization with TTOpt {comment}')

    def func(X):
        imgs = gen.visualize_batch_np(X)
        imgs = resize_and_pad(imgs, (227, 227), (0, 0))
        return model.run_target(imgs)

    tto = TTOpt(func, d=d, a=0., b=1., p=2, q=5, evals=evals,
        name='LATENT', with_cache=True, with_log=True)
    tto.maximize(rank)

    image = gen.render(tto.x_min.reshape((1, -1)), scale=1.0)[0]
    image.show('result/image/opt_ttopt.png')

    y = model.run(image)

    print(f'Activation in target neuron : {model.hook.a:-14.8e}')
    log.res(tpc() - _time)


def run_opt_am(model, image, evals, comment=''):
    """Run optimization with Activation Maximization (AM)."""
    _time = tpc()
    log.prc(f'Optimization with Activation Maximization {comment}')

    x = image.to_tens(model.device, batch=True)
    x = opt_am(model, x, evals=int(evals))

    image = Image(x, 'tens')
    image.show('result/image/opt_am.png')

    print(f'Activation in target neuron : {model.hook.a:-14.8e}')
    log.res(tpc() - _time)


def run_predict(data, model, image, comment=''):
    """Run model prediction."""
    _time = tpc()
    log.prc(f'Simple ANN prediction for given input {comment}')

    y = model.run(image)
    for lbl in sort_vector(y)[:3]:
        log(f'    {lbl[1]*100:-5.1f}% : {data.labels[lbl[0]]}')

    print(f'Activation in target neuron : {model.hook.a:-14.8e}')
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

    demo(device, log)
