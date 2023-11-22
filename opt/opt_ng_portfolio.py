import nevergrad as ng
import numpy as np
from time import perf_counter as tpc


def opt_ng_portfolio(func, d, n, m, is_max=True):
    """Portfolio method from nevergrad."""

    info = {'m': 0, 'i': None, 'y': None, 't': 0., 't0': tpc(),
        'ml': [], 'il': [], 'yl': []}

    def func_wrap(i):
        I = np.array(i).reshape(1, -1)
        y = func(I)
        ind_opt = np.argmax(y)

        i_cur = I[ind_opt, :]
        y_cur = y[ind_opt]

        info['m'] += len(y)
        info['t'] = tpc() - info['t0']

        is_new = info['y'] is None
        is_new = is_new or is_max and info['y'] < y_cur
        is_new = is_new or not is_max and info['y'] > y_cur
        if is_new:
            info['i'] = i_cur.copy()
            info['y'] = y_cur
            info['ml'].append(info['m'])
            info['il'].append(i_cur.copy())
            info['yl'].append(y_cur)

            text = f'portfolio > '
            text += f'm {info["m"]:-7.1e} | '
            text += f't {info["t"]:-9.3e} | '
            text += f'y {info["y"]:-11.4e}'
            print(text)

        return y[0]

    optimizer = ng.optimizers.Portfolio(
        parametrization=ng.p.TransitionChoice(range(n), repetitions=d),
        budget=int(m), num_workers=1)
    recommendation = optimizer.provide_recommendation()
    #print(recommendation)

    for _ in range(optimizer.budget):
        x = optimizer.ask()
        optimizer.tell(x, func_wrap(x.value) * (-1 if is_max else 1))

    text = f'portfolio > '
    text += f'm {info["m"]:-7.1e} | '
    text += f't {info["t"]:-9.3e} | '
    text += f'y {info["y"]:-11.4e} <<< DONE'
    print(text)

    return info['i'], info['y'], (info['ml'], info['il'], info['yl'])
