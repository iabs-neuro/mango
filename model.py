import numpy as np
from time import perf_counter as tpc
import torch


from image import Image


class Model:
    def __init__(self, device):
        """Класс-обертка для работы с ИНС.

        Args:
            device (torch.device): опциональное устройство для переноса объекта.

        """
        self.device = device

        self.set()
        self.set_labels()
        self.set_shape()

        self.probs = torch.nn.Softmax(dim=1)

    def run(self, img=None, X=None):
        """Вычисление предсказания ИНС.

        Args:
            img (Image): экземпляр класса графического изображения, подаваемый
                на вход ИНС.
            X (np.ndarray): массив numpy формы "[samples, dimensions]", причем
                картинка представлена во flatten формате (и размерности и
                каналы преобразованы в плоский одномерный массив, то есть
                dimensions = size x size x channels). Должно быть задано либо
                img, либо X.

        Returns:
            np.ndarray: полный выходной вектор ИНС (одномерный np.ndarray, если
            был передан "img", либо двумерный вектор со значениями для
            переданного батча "X").

        """
        if img is None:
            X = np.asarray(X)
            X = torch.from_numpy(X).float()
            X = X.to(self.device)
            X = Image.norm(X) # TODO: customize.
        else:
            X = img.to_tens(self.device, batch=True)

        with torch.no_grad():
            self.model.eval()
            y = self.model(X)
            y = self.probs(y)
            y = y if img is None else y[0, :]
            y = y.to('cpu').numpy()

        return y

    def set(self, model=None, name=None):
        """Задание модели ИНС.

        Args:
            model (torch.nn.Module): ИНС в pytorch формате.
            name (str): отображаемое имя модели. Если непосредственно модель
                (аргумент "model") не задана, то данное имя должно совпадать с
                именем архитектуры, присутствующим в хранилище моделей pytorch
                (при этом будет произведена загрузка соответствующей модели).

        """
        if model is None and name is not None:
            self.model = torch.hub.load('pytorch/vision', name, pretrained=True)
        else:
            self.model = model

        if self.model is not None and self.device is not None:
            self.model.to(self.device)

        self.name = name

    def set_labels(self, labels={}, name_data=''):
        """Задание выходных классов для ИНС.

        Args:
            labels (dict): описание классов для классификатора. Ключи
                соответствуют выходам ИНС, а значения - соответствующим именам
                классов.
            name_data (str): имя набора данных ("imagenet", "mnist" и т.п.).

        """
        self.labels = labels
        self.name_data = name_data

    def set_shape(self, sz=28, ch=1):
        """Задание параметров входных изображений.

        Args:
            sz (int): размер входного изображения (предполагается, что все
                изображения квадратные).
            ch (int): количество каналов входного изображения.

        """
        self.sz = sz
        self.ch = ch

    def set_target(self, layer, filter):
        """Задание целевого нейрона в ИНС для исследования.

        Args:
            layer (str): имя целевого слоя сети.
            filter (int): номер фильтра слоя сети (нумерация с нуля).

        """
        self.layer = self.model.features[layer]
        if type(self.layer) != torch.nn.modules.conv.Conv2d:
            raise ValueError('Мы работаем только со сверточными слоями')

        self.filter = filter
        if self.filter < 0 or self.filter >= self.layer.out_channels:
            raise ValueError('Указан несуществующий номер фильтра')
