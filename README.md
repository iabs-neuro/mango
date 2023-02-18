# neural_tensor_train


## Description

Software product for analysis of activations and specialization in artificial neural networks (ANN) with the tensor train (TT) decomposition.


## Installation

1. Install [python](https://www.python.org) (version 3.8; you may use [anaconda](https://www.anaconda.com) package manager);

2. Create a virtual environment:
    ```bash
    conda create --name neural_tensor_train python=3.8 -y
    ```

3. Activate the environment:
    ```bash
    conda activate neural_tensor_train
    ```

4. Install dependencies:
    ```bash
    pip install teneva==0.13.0 ttopt==0.5.0 protes==0.1.2 torch torchvision scikit-image matplotlib PyYAML jupyterlab "jax[cpu]==0.4.3" optax
    ```

5. Delete virtual environment at the end of the work (optional):
    ```bash
    conda activate && conda remove --name neural_tensor_train --all -y
    ```


## Usage

Run `python demo.py`, then see the outputs in the terminal and results in the `result` folder (`log` and `image` subfolders).

> Before starting the next calculation, you can completely delete or rename the `result` folder. A new `result` folder will be created automatically in this case.


## Authors

- [Andrei Chertkov](https://github.com/AndreiChertkov)
- [Nikita Pospelov](https://github.com/niveousdragon)
- [Maxim Beketov](https://github.com/bekemax)
