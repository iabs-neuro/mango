from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import requests
import subprocess
from time import perf_counter as tpc
from urllib.parse import urlencode


class Log:
    def __init__(self, fpath=None):
        self.fpath = fpath
        self.is_new = True

    def __call__(self, text):
        print(text)
        if self.fpath:
            with open(self.fpath, 'w' if self.is_new else 'a') as f:
                f.write(text + '\n')
        self.is_new = False

    def end(self):
        dt = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
        content = f'Work is finished ({tpc()-self.tm:-.2f} sec. total)'
        text = '\n\n' + '=' * 21 + ' ' + '-' * len(content) + '\n'
        text += f'[{dt}] {content}\n\n'
        self(text)

    def prc(self, content=''):
        self(f'\n.... {content}')
        return tpc()

    def res(self, t, content=''):
        self(f'DONE ({t:-9.2f} sec.) {content}')

    def title(self, content, info):
        self.tm = tpc()
        dt = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
        text = f'[{dt}] {content}'
        text += '\n' + '=' * 21 + ' ' + '-' * len(content) + '\n'
        text += info
        text += '=' * (22 + len(content)) + '\n'
        self(text)

    def wrn(self, content=''):
        self(f'WRN ! {content}')


def load_repo(url, fpath):
    # Run web robot:
    prc = subprocess.getoutput(f'cd {fpath} && git clone {url}')
    print(prc)


def load_yandex(url, fpath):
    link = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'
    link += urlencode(dict(public_key=url))
    link = requests.get(link).json()['href']
    with open(fpath, 'wb') as f:
        f.write(requests.get(link).content)


def plot_hist_am(a, title='', fpath=None, size=6, bins=100):
    fig = plt.figure(figsize=(size, size))
    n, bins, patches = plt.hist(a, bins, density=False, facecolor='g')
    plt.xlabel('Activations')
    plt.ylabel('Counts')
    plt.title(title)
    plt.grid(True)
    plt.savefig(fpath, bbox_inches='tight') if fpath else plt.show()
    plt.close(fig)


def plot_opt_conv(data, title='', fpath=None, size=7, m_min=None):
    colors = [
        '#040A1F', '#1144AA', '#09EA48', '#CE0071',
        '#FFF800', '#CE0071', '#FFB300', '#6A1C07', '#A30FCB', '#1C5628']
    marker = [
        'D', 's', '*', 'p',
        's', 'o', 'p', 'p', 'p']

    fig = plt.figure(figsize=(size, size))
    for i, (meth, info) in enumerate(data.items()):
        x, y = np.array(info[0]), np.array(info[2])
        if m_min is not None:
            ind = np.argmax(x > m_min)
            x = x[ind:]
            y = y[ind:]
        plt.plot(x, y, label=meth, marker=marker[i], markersize=8,
            linewidth=4 if i==0 else 3, color=colors[i])

    plt.legend(loc='best', frameon=True)
    plt.xlabel('Number of requests to model')
    plt.ylabel('Activation')
    plt.title(title)
    plt.grid(True)
    plt.semilogx()
    plt.semilogy()
    plt.savefig(fpath, bbox_inches='tight') if fpath else plt.show()
    plt.close(fig)


def sort_matrix(A, asc=True):
    I = np.unravel_index(np.argsort(A, axis=None), A.shape)
    I = [(I[0][k], I[1][k]) for k in range(A.size)]
    return I[::-1] if asc else I


def sort_vector(a, asc=True):
    return sorted(zip(range(len(a)), a), key=lambda item: item[1], reverse=asc)
