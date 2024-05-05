from .manager import MangoManager
import numpy as np
from os.path import join

data = 'cifar10'
gen = 'gan_sn'
model = 'sjsnn'
tlayer = 'layer4.0.sn1'
task = 'am'
kind = 'unit'
root = f'D:\\Projects\\mango_data\\SJ-SNN-T50\\SJ-SNN final\\{model}_result_{tlayer}'
model_path = 'C:\\Users\\admin\\PycharmProjects\\mango\\model\\snn_cifar10\\checkpoint_max_test_acc1_t50.pth'
opt_args = {
    'opt_budget': 20000,
    #'am_methods': ['TT', 'TT-s', 'TT-b', 'TT-exp'],
    'am_methods': ['TT-exp', 'TT-exp', 'TT-exp'],
    'track_opt_progress': False,
    'res_mode': 'best',
    'nrep': 1
}

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