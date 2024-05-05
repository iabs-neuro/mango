from .manager import MangoManager
import os
from os.path import join
import shutil
import torch
import numpy as np
import tqdm

data = 'cifar10'
gen = 'gan_sn'
model = 'sjsnn'
task = 'scan'
kind = 'unit'
root = f'D:\\Projects\\mango_data\\Activity'
model_path = ("C:\\Users\\admin\\PycharmProjects\\logs_t50\\pt\\"
              "spiking_resnet18_b300_e1000_sgd_lr0.2_wd0.0_ls0.1_ma0.2_ca1.0_sbn0_ra0_re0.1_aaugta_wide_size32_232_224_seed2020_T50\\"
              "checkpoint_100.pth")

opt_args = {}

ALL_SN_LAYERS = ['sn1', 'layer1.0.sn1', 'layer1.0.sn2', 'layer1.1.sn1', 'layer1.1.sn2', 'layer2.0.sn1', 'layer2.0.sn2',
                 'layer2.1.sn1', 'layer2.1.sn2', 'layer3.0.sn1', 'layer3.0.sn2', 'layer3.1.sn1', 'layer3.1.sn2',
                 'layer4.0.sn1', 'layer4.0.sn2', 'layer4.1.sn1', 'layer4.1.sn2']

ITERS = [0,5,10,20,30,40,50,60,70,80,90,100,125,150,175,200,225,250,275,300,400,500,600,700,800,900]

for iter in ITERS:
    model_path = ("C:\\Users\\admin\\PycharmProjects\\logs_t50\\pt\\"
                  "spiking_resnet18_b300_e1000_sgd_lr0.2_wd0.0_ls0.1_ma0.2_ca1.0_sbn0_ra0_re0.1_aaugta_wide_size32_232_224_seed2020_T50\\"
                  f"checkpoint_{iter}.pth")

    manager = MangoManager(
        data=data,
        gen=gen,
        model=model,
        task=task,
        kind='layer',
        opt_args=opt_args,
        root=root,
        model_path=model_path,
        layer_names=ALL_SN_LAYERS)

    manager.run()

    ###############################  TRAIN #################################
    batch_size = 500
    chunks = np.array_split(np.arange(50000), 50000//batch_size)

    all_activity = {lname: [] for lname in ALL_SN_LAYERS}

    for k, chunk in tqdm.tqdm(enumerate(chunks), total=len(chunks)):
        x_real = torch.cat([manager.data.get(i=_)[0][None] for _ in chunk])
        y = manager.model.run(x_real)

        for j, hr in enumerate(manager.model.hook_result):
            lname = ALL_SN_LAYERS[j]
            all_activity[lname].append(hr.cpu())

        manager.model.hook_result = []

    actdir = os.path.join(root, f'iter {iter}', 'train')
    os.makedirs(actdir, exist_ok=True)
    for _, lname in tqdm.tqdm(enumerate(ALL_SN_LAYERS)):
        act_data = torch.cat(all_activity[lname], dim=1)
        act_arr = act_data.numpy()
        np.savez_compressed(os.path.join(actdir, f'SJ-SNN act iter {iter} train layer {lname}.npz'), a=act_arr)

    ############################### TEST #################################
    batch_size = 500
    chunks = np.array_split(np.arange(10000), 10000 // batch_size)

    all_activity = {lname: [] for lname in ALL_SN_LAYERS}

    for k, chunk in tqdm.tqdm(enumerate(chunks), total=len(chunks)):
        x_real = torch.cat([manager.data.get(i=_, tst=True)[0][None] for _ in chunk])
        y = manager.model.run(x_real)

        for j, hr in enumerate(manager.model.hook_result):
            lname = ALL_SN_LAYERS[j]
            all_activity[lname].append(hr.cpu())

        manager.model.hook_result = []

    actdir = os.path.join(root, f'iter {iter}', 'test')
    os.makedirs(actdir, exist_ok=True)
    for _, lname in tqdm.tqdm(enumerate(ALL_SN_LAYERS)):
        act_data = torch.cat(all_activity[lname], dim=1)
        act_arr = act_data.numpy()
        np.savez_compressed(os.path.join(actdir, f'SJ-SNN act iter {iter} test layer {lname}.npy'), act_arr)
