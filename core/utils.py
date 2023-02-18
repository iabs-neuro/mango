from datetime import datetime
import matplotlib.pyplot as plt
import os
from skimage.transform import resize
import torch
import torch.nn.functional as F


class Log:
    def __init__(self, fpath=None):
        self.fpath = fpath
        self.is_new = True
        self.len_pref = 10

    def __call__(self, text):
        print(text)
        if self.fpath:
            with open(self.fpath, 'w' if self.is_new else 'a') as f:
                f.write(text + '\n')
        self.is_new = False

    def prc(self, content=''):
        self(f'\n.... {content}')

    def res(self, t, content=''):
        self(f'DONE ({t:-9.2f} sec.) {content}')

    def title(self, content):
        dt = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
        text = f'[{dt}] {content}'
        text += '\n' + '=' * 21 + ' ' + '-' * len(content) + '\n'
        self(text)


def folder_ensure(fpath):
    os.makedirs(fpath, exist_ok=True)


def plot_image(img, fpath=None):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    ax.imshow(img)
    ax.axis('off')

    if fpath:
        plt.savefig(fpath, bbox_inches='tight')
    else:
        plt.show()


def resize_and_pad(img_tsr, size, offset, canvas_size=(227, 227)):
    # image is in (0,1) scale so padding with 0.5 as gray background.
    assert img_tsr.ndim in [3, 4]
    if img_tsr.ndim == 3:
        img_tsr.unsqueeze_(0)
    imgn = img_tsr.shape[0]

    padded_shape = (imgn, 3) + canvas_size
    pad_img = torch.ones(padded_shape) * 0.5
    pad_img.to(img_tsr.dtype)

    rsz_tsr = F.interpolate(img_tsr, size=size)
    pad_img[:, :, offset[0]:offset[0] + size[0], offset[1]:offset[1] + size[1]] = rsz_tsr

    return pad_img


def sort_vector(a, asc=True):
    return sorted(zip(range(len(a)), a), key=lambda item: item[1], reverse=asc)
