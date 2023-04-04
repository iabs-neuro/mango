import numpy as np
from ttopt import TTOpt


def opt_ttopt(func, d, n, m, rank=5, with_qtt=False, is_max=True):
    """Activation maximization with TTOpt."""
    if with_qtt:
        q = np.log2(n)
        if 2**q != n:
            raise ValueError('Invalid grid size. It should be power of 2')

    info = {'m': 0, 'i': None, 'y': None, 'ml': [], 'il': [], 'yl': []}

    def func_wrap(I):
        y = func(I)
        ind_opt = np.argmax(y)

        i_cur = I[ind_opt, :]
        y_cur = y[ind_opt]

        info['m'] += len(y)

        is_new = info['y'] is None
        is_new = is_new or is_max and info['y'] < y_cur
        is_new = is_new or not is_max and info['y'] > y_cur
        if is_new:
            info['i'] = i_cur.copy()
            info['y'] = y_cur
            info['ml'].append(info['m'])
            info['il'].append(i_cur.copy())
            info['yl'].append(y_cur)

        return y

    tto = TTOpt(func_wrap, d=d,
        n=None if with_qtt else n,
        p=2 if with_qtt else None,
        q=q if with_qtt else None,
        evals=m, name='ttopt', is_func=False, with_cache=True, with_log=True)
    if is_max:
        tto.maximize(rmax=rank)
    else:
        tto.minimize(rmax=rank)

    return info['i'], info['y'], (info['ml'], info['il'], info['yl'])
