import os
from os.path import join
import pickle

data = 'cifar10'
gen = 'gan_sn'
model = 'sjsnn'
layer = 'layer1.0.sn2'

folder = f'{model}-{data}-{gen}-{layer}_processed'
imagefolder = join(folder, 'AM_images')
datafolder = join(folder, 'dat')
datafiles = os.listdir(datafolder)

for pdata in datafiles[:1]:
    print(pdata)
    with open(join(datafolder, pdata), 'rb') as f:
        data = pickle.load(f)
        print(data['TT-b'])

