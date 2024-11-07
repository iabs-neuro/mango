from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.ticker
import numpy as np
import requests
import subprocess
from time import perf_counter as tpc
from urllib.parse import urlencode

import os
import sys
import time

import torch.nn as nn
import torch.nn.init as init

class Log:
    def __init__(self, fpath=None):
        self.fpath = fpath
        self.is_new = True

    def __call__(self, log_input):
        if not isinstance(log_input, str):
            text = repr(log_input)
        else:
            text = log_input

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


def plot_opt_conv(data, title='', fpath=None, size=18, m_min=None):
    colors = [
        '#040A1F', '#1144AA', '#09EA48', '#CE0071',
        '#FFF800', '#CE0071', '#FFB300', '#6A1C07', '#A30FCB', '#1C5628']
    marker = [
        'D', 's', '*', 'p',
        's', 'o', 'p', 'p', 'p']

    fig, ax = plt.subplots(figsize=(size, size))
    ax = make_beautiful(ax)

    ax.set_xscale('log')
    ax.set_yscale('log')

    for i, (meth, info) in enumerate(data.items()):
        x, y = np.array(info[1]), np.array(info[3])
        if m_min is not None:
            ind = np.argmax(x > m_min)
            x = x[ind:]
            y = y[ind:]

        ax.plot(x, y, label=meth, marker=marker[i], markersize=20,
            linewidth=6 if i == 0 else 5, color=colors[i])

    #ax.set_xticks([0.1, 0.2, 0.3, 0.4])
    #ax.set_yticks([0.1, 0.2, 0.3, 0.4])
    ax.legend(loc='best', frameon=True)
    ax.set_xlabel('Number of requests to model')
    ax.set_ylabel('Activation')
    ax.set_title(title)
    ax.grid(visible=True, axis='both')
    fig.savefig(fpath, bbox_inches='tight') if fpath else plt.show()
    plt.close(fig)


def sort_matrix(A, asc=True):
    I = np.unravel_index(np.argsort(A, axis=None), A.shape)
    I = [(I[0][k], I[1][k]) for k in range(A.size)]
    return I[::-1] if asc else I


def sort_vector(a, asc=True):
    return sorted(zip(range(len(a)), a), key=lambda item: item[1], reverse=asc)


def make_beautiful(ax):
    for axis in ['bottom', 'left']:
        ax.spines[axis].set_linewidth(4)
        ax.tick_params(width=4, direction='in', length=8, pad=15)

    for axis in ['top', 'right']:
        ax.spines[axis].set_linewidth(1.0)

    #ax.locator_params(axis='x', nbins=8)
    #ax.locator_params(axis='y', nbins=8)
    ax.tick_params(axis='both', which='major', labelsize=26)
    ax.tick_params(axis='both', which='minor', labelsize=26)

    ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    ax.yaxis.set_major_locator(plt.MaxNLocator(10))

    ax.xaxis.label.set_size(30)
    ax.yaxis.label.set_size(30)

    params = {'legend.fontsize': 30,
              'axes.titlesize': 40,
              }

    pylab.rcParams.update(params)

    return ax


'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


#_, term_width = os.popen('stty size', 'r').read().split()
term_width = 90.0
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f