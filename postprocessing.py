
import numpy as np
import tqdm
import os
import pandas as pd
from os.path import join
from .manager import MangoManager
from scipy.stats import entropy
import torch
from PIL import Image
import yaml

root = f'D:\\Projects\\mango_data\\SJ-SNN-T50'

#LAYERS = ['sn1', 'layer1.0.sn2', 'layer1.1.sn2', 'layer2.0.sn2', 'layer2.1.sn2',
#          'layer3.0.sn2', 'layer3.1.sn2', 'layer4.0.sn2', 'layer4.1.sn2', 'fc']
LAYERS = ['sn1', 'layer1.0.sn1', 'layer1.1.sn1', 'layer2.0.sn1', 'layer2.1.sn1',
'layer3.0.sn1', 'layer3.1.sn1', 'layer4.0.sn1', 'layer4.1.sn1']

gen = 'gan_sn'
#METHODS = ['TT', 'TT-s', 'TT-b']
METHODS = ['0_TT-exp', '1_TT-exp', '2_TT-exp']

ITERS = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 400, 500, 600, 700]

dataname = 'cifar10'
model = 'sjsnn'
task = 'am'
kind = 'unit'
opt_args = {
    'opt_budget': 20000,
    'am_methods': METHODS
}
units = np.arange(64)

nrand = 25

reload_from_pickles = 0
all_data = {iter_: dict() for iter_ in ITERS}

for iter_ in tqdm.tqdm(ITERS[:], position=0):
    # ======================  upload model for this iter  ==========================
    model_path = f"D:\\Projects\\mango_data\\logs_t50\\checkpoint_{iter_}.pth"

    # we do not perform AM and use the network for inference only, manager params have no effect (except for model_path)
    manager = MangoManager(data=dataname,
                           gen=gen,
                           model=model,
                           task=task,
                           kind=kind,
                           cls=None,
                           unit=0,
                           layer='sn1',
                           opt_args=opt_args,
                           root=root,
                           model_path=model_path)

    # ======================  upload opt data  ==========================
    all_data[iter_] = {layer: dict() for layer in LAYERS}

    for layer in tqdm.tqdm(LAYERS[:], leave=True, position=0):
        all_data[iter_][layer] = {unit: dict() for unit in units}

        folder = os.path.join(root, f'SJ-SNN iter {iter_}', f'{model}-{dataname}-gan_sn-{layer}_processed')
        imagefolder = join(folder, 'AM_images')
        datafolder = join(folder, 'dat')
        datafiles = os.listdir(datafolder)

        for unit in units[:]:
            all_data[iter_][layer][unit] = {method: dict() for method in METHODS}

            pdata = f'opt_info_u{unit}_{layer}.pkl'
            try:
                with open(join(datafolder, pdata), 'rb') as f:
                    #data = pickle.load(f) # ==================== super slow
                    data = pd.read_pickle(f)
                    #print(data)
                    for method in METHODS:
                        # ======================  upload computed opt data  ==========================
                        tvals, zvals, activations = data[method]
                        if len(tvals) == 0:
                            opt_info = {
                                'max_t': 0,
                                'max_z': np.zeros(128),
                                'max_a': 0
                            }
                        else:
                            opt_info = {
                                'max_t': tvals[-1],
                                'max_z': zvals[-1],
                                'max_a': activations[-1]
                            }

                        all_data[iter_][layer][unit][method] = opt_info

                        # ======================  augment opt data  ==========================
                        latent = all_data[iter_][layer][unit][method]['max_z']
                        z = manager.gen.ind_to_poi(latent)
                        x = manager.gen.run(z)

                        all_probs = np.zeros((nrand, 10))
                        rel_perpl = np.zeros(nrand)
                        for i in range(nrand):
                            res = manager.model.run(x)

                            probs_fn = torch.nn.Softmax(dim=0)
                            probs = probs_fn(res).cpu().numpy()
                            all_probs[i, :] = probs
                            H0 = 2 ** (entropy([0.1 for _ in range(10)]))
                            H = 2 ** entropy(probs)

                            rel_perpl[i] = H / H0

                        max_image = manager.model.x_to_image(x.cpu().numpy())
                        all_data[iter_][layer][unit][method]['probs'] = np.mean(all_probs, axis=0)
                        all_data[iter_][layer][unit][method]['rel_perpl'] = np.mean(rel_perpl)
                        all_data[iter_][layer][unit][method]['max_image'] = max_image

                        # save .jpeg for further complexity analysis
                        im = Image.fromarray((max_image * 255).astype(np.uint8))
                        os.makedirs(join(folder, 'JPEG images'), exist_ok=True)
                        jpeg_path = join(folder, 'JPEG images', f"am_u{unit}_{layer}_{method}.jpeg")
                        im.save(jpeg_path)
                        all_data[iter_][layer][unit][method]['complexity'] = os.path.getsize(jpeg_path)

            except FileNotFoundError:
                print(f'not found: {join(datafolder, pdata)}')
                pass

    with open(f'D:\\Projects\\mango_data\\SJ-SNN evolution data full 3 (after iter {iter_}).yaml', 'w') as f:
        yaml.dump(all_data, f)
