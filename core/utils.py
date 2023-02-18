from datetime import datetime
import os


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


def sort_vector(a, asc=True):
    return sorted(zip(range(len(a)), a), key=lambda item: item[1], reverse=asc)
