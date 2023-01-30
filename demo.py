import numpy as np
import random
import sys
from time import perf_counter as tpc
import torch
from ttopt import TTOpt


from am import am
from data import Data
from gen_gan import GenGAN
from image import Image
from model import Model
from model_wrapper import ModelWrapper
from utils import folder_ensure
from utils import plot_image
from utils import resize_and_pad
from utils import sort_vector


def run_am():
    _time = tpc()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    data = Data('imagenet')
    data.load_labels()

    model = Model(device)
    model.set(name='vgg16')
    model.set_labels(data.labels, data.name)
    model.set_shape(data.sz, data.ch)
    model.set_target(layer=2, filter=10)

    image = Image('demo/demo_image1.jpg', 'file')

    # Запуск предсказания ИНС (просто для интереса):
    y = model.run(image)
    print(f'\n\n\n>>>ANN predictions for given image:')
    for lbl in sort_vector(y)[:3]:
        print(f'{lbl[1]*100:-5.1f}% : ', data.labels[lbl[0]])

    # Можно взять реальное изображение в качестве базового, либо случайное:
    # image_base = Image.rand(data.sz, data.ch, 50, 180)
    image_base = image

    image_am = am(model, image_base, iters=50)
    image_am.show('result/am.png')

    print(f'\n\nDONE | Time: {tpc() - _time:-10.3f} sec.')


def run_gan(evals=1.E+3, rank=4):
    _time = tpc()

    model = ModelWrapper('alexnet', device='cpu')
    model.select_unit(('alexnet', '.classifier.Linear6', 1))
    gen = GenGAN(name='fc8')
    d = gen.codelen

    def func(X):
        imgs = gen.visualize_batch_np(X)
        imgs = resize_and_pad(imgs, (227, 227), (0, 0))
        return model.score_tsr(imgs)

    def callback(last):
        if True: # Only for debug (plot every result improvement)
            img = gen.render(tto.x_min.reshape((1, -1)), scale=1.0)[0]
            plot_image(img, 'result/gan_ttopt.png')

    tto = TTOpt(func, d=d, a=0., b=1., p=2, q=5, evals=evals,
        callback=callback, name='LATENT', with_cache=True, with_log=True)
    tto.maximize(rank)

    img = gen.render(tto.x_min.reshape((1, -1)), scale=1.0)[0]
    plot_image(img, 'result/gan_ttopt.png')

    print(f'\n\nDONE | Time: {tpc() - _time:-10.3f} sec.')


if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    folder_ensure('result')
    folder_ensure('result/gan')
    folder_ensure('result/logs')

    mode = sys.argv[1] if len(sys.argv) > 1 else 'am'

    if mode == 'am':
        run_am()
    elif mode == 'gan':
        run_gan()
    else:
        raise ValueError(f'Invalid computation mode "{mode}"')
