from datetime import datetime
import os
import torch


class Manager:
    def __init__(self, name='', device=None):
        self.name = name
        self.set_device(device)
        self.set_path()
        self.load()

    def get_path(self, fpath):
        return os.path.join(self.path_result, fpath)

    def load(self):
        raise NotImplementedError('Must be set in the child class')

    def run(self):
        raise NotImplementedError('Must be set in the child class')

    def set_device(self, device=None):
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device

    def set_path(self, root_result='result'):
        self.path_root = os.path.dirname(__file__) + '/..'
        self.path_result = os.path.join(self.path_root, root_result)
        os.makedirs(self.path_result, exist_ok=True)

        if self.name:
            self.path_result = os.path.join(self.path_result, self.name)
            os.makedirs(self.path_result, exist_ok=True)

    def tmp1(self):
        #model = Model('vgg16', data)
        data = Data('imagenet')
        x = data.img_load('demo_image.jpg')
        print(x.shape)
        data.plot(x, 'Transformed', fpath=f'tmp.png')

    def tmp2(self):
        data = Data('cifar10')
        x, c = data.get(42)
        data.plot(x, fpath='tmp.jpg')
        x = data.img_load('demo_image.jpg')
        data.plot(x, 'Transformed', fpath=f'tmp.jpg')

    def tmp3(self):
        samples = 25
        z = torch.randn(samples, gen.d).to(device)
        x = gen.run(z)
        y = model.run(x)
        p = torch.argmax(y, axis=1).detach().to('cpu').numpy()
        l = [data.labels[p_cur] for p_cur in p]
        print(l)

        data.plot_many(x, l, cols=5, rows=5, fpath=f'result_tmp/gen_random.png')


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


def sort_vector(a, asc=True):
    return sorted(zip(range(len(a)), a), key=lambda item: item[1], reverse=asc)
