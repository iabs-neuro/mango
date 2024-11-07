import os

from .manager import MangoManager
import numpy as np
from os.path import join
from PIL import Image
import shutil


data = 'cifar10'
gen = 'gan_sn'
model = 'sjsnn'
tlayer = 'layer4.1.sn1'
task = 'am'
kind = 'unit'
iter_ = 10
root = f'D:\\Projects\\mango_data\\SJ-SNN-T50\\SJ-SNN iter {iter_}\\{model}_result_{tlayer}'
model_path = f"D:\Projects\mango_data\logs_t50\checkpoint_{iter_}.pth"
opt_args = {
    'opt_budget': 20000,
    #'am_methods': ['TT', 'TT-s', 'TT-b', 'TT-exp'],
    'am_methods': ['TT-exp', 'TT-exp', 'TT-exp'],
    'track_opt_progress': False,
    'res_mode': 'best',
    'nrep': 1
}

'''
manager = MangoManager(
    data=data,
    gen=gen,
    model=model,
    task=task,
    kind=kind,
    cls=None,
    unit=0,
    layer=tlayer,
    opt_args=opt_args,
    root=root,
    model_path=model_path
)

labels = np.array([manager.data.get(i=_, tst=0)[1] for _ in range(50000)])
np.savez(join('D:\\Projects\\mango_data', 'CIFAR train labels'), labels)
'''

'''
imroot = join(os.path.dirname(root), f'{model}-{data}-{gen}-{tlayer}_processed')
impath = join(imroot, 'AM_images')
imnames = os.listdir(impath)

for imname in imnames:
    im = Image.open(join(impath, imname))
    rgb_im = im.convert('RGB')
    os.makedirs(join(imroot, 'AM_images_jpg'), exist_ok=True)
    rgb_im.save(join(imroot, 'AM_images_jpg', imname[:-4] + '.jpg'), quality=10)
'''


def process_results(data, gen, root, model, layer):
    '''
    Copies layer MEI results to a separate folder with more convenient structure
    '''
    new_folder = join(os.path.dirname(root), f'{model}-{data}-{gen}-{layer}_processed')
    os.makedirs(new_folder, exist_ok=True)
    for dtype in ['AM_images', 'opt_conv', 'opt_conv_short', 'log', 'gif', 'dat']:
        os.makedirs(join(new_folder, dtype), exist_ok=True)

    signature = f'{data}-{gen}-{model}'
    mei_res_folders = os.listdir(join(root, signature))

    for folder in mei_res_folders:
        path = join(root, signature, folder)
        try:
            images = os.listdir(join(path, 'img'))

            # extract am signature from image files
            for im in images:
                if 'conv' not in im:
                    imsig = im[3:-4]

            for im in images:
                if imsig not in im:
                    new_imname = im[:-4] + '_' + imsig + im[-4:]
                else:
                    new_imname = im

                if 'opt_conv' in im:
                    shutil.copy(join(path, 'img', im), join(new_folder, 'opt_conv', new_imname))
                if 'opt_conv_short' in im:
                    shutil.copy(join(path, 'img', im), join(new_folder, 'opt_conv_short', new_imname))
                if 'am' in im:
                    shutil.copy(join(path, 'img', im), join(new_folder, 'AM_images', new_imname))

            if opt_args['track_opt_progress']:
                gifs = os.listdir(join(path, 'gif'))
                for gif in gifs:
                    shutil.copy(join(path, 'gif', gif), join(new_folder, 'gif', gif))

            dat = os.listdir(join(path, 'dat'))
            for dt in dat:
                new_dtname = dt[:-4] + '_' + imsig + dt[-4:]
                shutil.copy(join(path, 'dat', dt), join(new_folder, 'dat', new_dtname))

            new_logname = 'log_' + imsig + '.txt'
            shutil.copy(join(path, 'log.txt'), join(new_folder, 'log', new_logname))

        except FileNotFoundError:
            print(f'{folder} not found in results')

LAYERS = ['sn1', 'layer1.0.sn1', 'layer1.1.sn1', 'layer2.0.sn1', 'layer2.1.sn1',
'layer3.0.sn1', 'layer3.1.sn1', 'layer4.0.sn1', 'layer4.1.sn1']

for tlayer in LAYERS:
    root = f'D:\\Projects\\mango_data\\SJ-SNN-T50\\opt full2\\{model}_result_{tlayer}'
    process_results(data, gen, root, model, tlayer)