# neural_tensor_train


## Description

Software product for analysis of activations and specialization in artificial neural networks (ANN) with the tensor train (TT) decomposition.


## Installation

1. Install [python](https://www.python.org) (version 3.7; you may use [anaconda](https://www.anaconda.com) package manager);

2. Create a virtual environment:
    ```bash
    conda create --name neuro_tensor_train python=3.7 -y
    ```

3. Activate the environment:
    ```bash
    conda activate neuro_tensor_train
    ```

4. Install dependencies:
    ```bash
    pip install teneva==0.12.8 ttopt==0.5.0 torch torchvision scikit-image matplotlib PyYAML jupyterlab
    ```

5. Delete virtual environment at the end of the work (optional):
    ```bash
    conda activate && conda remove --name neuro_tensor_train --all -y
    ```


## Usage

1. Perform the steps from `Installation` section

2. Run `python demo_ttopt.py`, then see the outputs in the terminal

3. Run `python demo.py am`, then see the outputs in the terminal and the `result` folder with the result for simple Activation Maximization (AM)

4. Run `python demo.py gan`, then see the outputs in the terminal and the `result` folder with the result for real test


## Usage with colab

TODO: сделать загрузку приватного репозитория в COLAB и демонстрационный код с запуском основных методов (перспективно, т.к. в colab есть GPU)


## Authors

- [Andrei Chertkov](https://github.com/AndreiChertkov)
- [Nikita Pospelov](???)
- [Maxim Beketov](https://github.com/bekemax)
