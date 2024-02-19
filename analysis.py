from .manager import MangoManager
import os
from os.path import join
import shutil

data = 'cifar10'
gen = 'gan-sn'
model = 'sjsnn'
tlayer = 'sn1'
task = 'am'
kind = 'unit'
root = f'{model}_result_{tlayer}'
opt_args = {
    'opt_budget': 12000,
    'am_methods': ['TT', 'TT-s', 'TT-b']
}


for i in range(64):
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
        root=root
    )
    manager.run()


def process_results(data, gen, root, model, layer):
    '''
    Copies layer MEI results to a separate folder with more convenient structure
    '''
    new_folder = f'{model}-{data}-{gen}-{layer}_processed'
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

        gifs = os.listdir(join(path, 'gif'))
        for gif in gifs:
            shutil.copy(join(path, 'gif', gif), join(new_folder, 'gif', gif))

        dat = os.listdir(join(path, 'dat'))
        for dt in dat:
            new_dtname = dt[:-4] + '_' + imsig + dt[-4:]
            shutil.copy(join(path, 'dat', dt), join(new_folder, 'dat', new_dtname))

        new_logname = 'log_' + imsig + '.txt'
        shutil.copy(join(path, 'log.txt'), join(new_folder, 'log', new_logname))


process_results(data, gen, root, model, tlayer)
