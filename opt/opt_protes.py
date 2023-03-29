from protes import protes


def opt_protes(func, d, n, m, is_max=True):
    """Activation Maximization with PROTES."""

    info = {}
    i, y = protes(func, d, n, m, is_max=is_max, log=True,
        info=info, with_info_i_opt_list=True)

    ml = info['m_opt_list']
    il = info['i_opt_list']
    yl = info['y_opt_list']

    return i, y, (ml, il, yl)
