import numpy as np
from protes import protes
import teneva


def opt_protes(func, d, n, m, k=50, k_top=5, with_qtt=False, is_max=True):
    """Activation Maximization with PROTES."""

    q = int(np.log2(n))
    if with_qtt and 2**q != n:
        raise ValueError('Invalid grid size. It should be power of 2 for QTT')

    def func_qtt(I_qtt):
        I_qtt = np.array(I_qtt, dtype=int)
        return func(teneva.ind_qtt_to_tt(I_qtt, q))

    info = {}
    i, y = protes(
        func_qtt if with_qtt else func,
        d * q if with_qtt else d,
        2 if with_qtt else n,
        m, k=k, k_top=k_top,
        is_max=is_max, log=True, info=info, with_info_i_opt_list=True)

    ml = info['m_opt_list']
    il = info['i_opt_list']
    yl = info['y_opt_list']

    if with_qtt:
        i = teneva.ind_qtt_to_tt(np.array(i, dtype=int), q)
        for j in range(len(il)):
            il[j] = teneva.ind_qtt_to_tt(np.array(il[j], dtype=int), q)

    return i, y, (ml, il, yl)
