import numpy as np
import random
import sys
import torch


from manager import ManagerDensenetCheck
from manager import ManagerDensenetOut


def run(task):
    # TODO: this is a draft
    man = ManagerDensenetOut()
    man.run()


if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    task = sys.argv[1] if len(sys.argv) > 1 else 'all'

    run(task)
