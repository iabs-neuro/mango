from .manager import MangoManager
import os
from os.path import join
import shutil
import numpy as np

data = 'cifar10'
gen = 'gan_sn'
model = 'resnet18'
tlayer = 'layer1.1'
task = 'am'
kind = 'unit'
iter_ = 97
#root = f'D:\\Projects\\mango_data\\SJ-SNN-T50\\SJ-SNN iter {iter_}\\{model}_result_{tlayer}'
#root = f'D:\\Projects\\mango_data\\SJ-SNN-T50\\opt full2\\{model}_result_{tlayer}'
root = f'D:\\Projects\\mango_data\\ResNet18'
#model_path = f"D:\Projects\mango_data\logs_t50\checkpoint_{iter_}.pth"
#model_path = f"D:\\Projects\\mango_data\\resnet18_logs\\checkpoint_{iter_}.pth"
model_path = None

opt_args = {
    'opt_budget': 1000,
    'am_methods': ['RS', 'NG', 'TT', 'TT-s', 'TT-b', 'TT-exp'],
    #'am_methods': ['RS', 'NG', 'TT', 'TT-s', 'TT-b', 'TT-exp'],#, 'TT-exp', 'TT-exp'],
    #'am_methods': ['RS', 'NG', 'TT-exp'], #, 'TT-exp'],
    'track_opt_progress': True,
    'res_mode': 'best',
    'nrep': 1
}

units_to_scan = np.arange(10, 11)
for i in units_to_scan:
    try:
        manager = MangoManager(
            data=data,
            gen=gen,
            model=model,
            task=task,
            kind=kind,
            cls=None,
            unit=i,
            layer=tlayer,
            opt_args=opt_args,
            root=root,
            model_path=model_path
        )
        manager.run()
    except TypeError: # rare nevergrad failure
        pass


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


#process_results(data, gen, root, model, tlayer)
