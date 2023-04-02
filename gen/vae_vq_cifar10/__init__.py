"""VQ-VAI on CIFAR10.

We use discrete VAE architecture from the work "Neural Discrete Representation
Learning" (https://arxiv.org/abs/1711.00937). The code is prepared based on the
example from colab (https://colab.research.google.com/github/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb).

"""
from .vae_vq_cifar10 import VAEVqCifar10
