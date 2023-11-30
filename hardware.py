import multiprocessing
import os


def configure_hardware(verbose=False):
    # attempts multiprocessing
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count={}".format(
        multiprocessing.cpu_count()
    )
    if verbose:
        print('cpu count:', multiprocessing.cpu_count())

    import jax
    # For faster and more accurate PROTES optimizer:
    jax.config.update('jax_enable_x64', True)
    # os.environ['JAX_PLATFORM_NAME'] = 'cpu'
    jax.config.update('jax_platform_name', 'cpu')

    if verbose:
        platform = jax.lib.xla_bridge.get_backend().platform.casefold()
        print("Platform: ", platform)
