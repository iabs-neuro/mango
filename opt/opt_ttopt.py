from ttopt import TTOpt


def opt_ttopt():

    def func(X):
        imgs = gen.visualize_batch_np(X)
        imgs = resize_and_pad(imgs, (227, 227), (0, 0))
        return model.run_target(imgs)

    tto = TTOpt(func, d=d, a=0., b=1., p=2, q=5, evals=evals,
        name='LATENT', with_cache=True, with_log=True)
    tto.maximize(rank)

    return
