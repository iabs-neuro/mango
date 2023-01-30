import numpy as np
import random
import sys
from time import perf_counter as tpc
import torch


from am import am
from data import Data
from image import Image
from model import Model
from utils import folder_ensure
from utils import sort_vector


def run_am():
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

    am_time = tpc()
    image_am = am(model, image_base, iters=50)
    image_am.show('result/am.png')
    am_time = tpc() - am_time
    print(f'\n\n\n>>>Activation Maximization is built (time={am_time:-8.5f})')


if __name__ == '__main__':
    np.random.seed(42)
    random.seed(42)
    torch.manual_seed(42)

    folder_ensure('result')
    folder_ensure('result/logs')

    mode = sys.argv[1] if len(sys.argv) > 1 else 'am'

    if mode == 'am':
        run_am()
    else:
        raise ValueError(f'Invalid computation mode "{mode}"')
