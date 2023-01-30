import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from PIL import Image as ImagePIL
import requests
import torch
import torchvision


class Image:
    @staticmethod
    def is_pict(pict):
        return isinstance(pict, ImagePIL.Image)

    @staticmethod
    def norm(tens, m=[0.485, 0.456, 0.406], v=[0.229, 0.224, 0.225]):
        return torchvision.transforms.Compose([
            torchvision.transforms.Normalize(m, v)
        ])(tens)

    @staticmethod
    def norm_back(tens, m=[0.485, 0.456, 0.406], v=[0.229, 0.224, 0.225]):
        for t, m, s in zip(tens, m, v):
            t.mul_(s).add_(m)
        return tens

    @staticmethod
    def rand(sz=224, ch=3, a=0, b=255):
        ndar = np.uint8(np.random.uniform(a, b, (ch, sz, sz)))
        return Image(ndar, 'ndar', sz, ch)

    def __init__(self, data=None, kind='data', sz=224, ch=3, norm_back=False):
        """Класс-обертка для работы с графическим изображением.

        Args:
            data (any): данные, по которым строится изображение.
            kind (str): тип данных:
                "data" (byte): непосредственно изображение в raw формате;
                "link" (str): url-адрес изображения;
                "tens" (torch.tensor): изображение в форме pytorch-тензора;
                "pict" (PIL.Image): изображение в PIL-формате;
                "ndar" (np.ndarray): изображение в форме numpy-массива;
                "file" (str): путь к файлу с изображением.
            sz (int): размер входного изображения (предполагается, что все
                изображения квадратные).
            ch (int): количество каналов входного изображения.
            norm_back (bool): если флаг задан, то для изображения будет
                проведена "антинормировка".

        """
        self.sz = sz
        self.ch = ch
        self.m = [0.485, 0.456, 0.406]
        self.v = [0.229, 0.224, 0.225]

        if kind == 'data':
            self.pict = ImagePIL.open(BytesIO(data))
        elif kind == 'link':
            response = requests.get(data)
            self.pict = ImagePIL.open(BytesIO(response.content))
        elif kind == 'tens':
            self.pict = torchvision.transforms.functional.to_pil_image(data)
        elif kind == 'pict':
            self.pict = data.copy()
        elif kind == 'ndar':
            data = torch.from_numpy(data).float()
            self.pict = torchvision.transforms.functional.to_pil_image(data)
        elif kind == 'file':
            self.pict = ImagePIL.open(data)
        else:
            raise ValuerError('Image is not set')

        self.pict_raw = self.pict.copy()

        tens = self.to_tens(norm=False)
        if norm_back:
            tens = Image.norm_back(tens, self.m, self.v)
        self.pict = torchvision.transforms.functional.to_pil_image(tens)

    def copy(self):
        return Image(self.pict, 'pict', self.sz, self.ch)

    def show(self, fpath=None):
        """Отрисовка изображения и сохранение в файл."""
        fig = plt.figure(figsize=(6, 6))

        plt.imshow(self.to_tens(norm=False).permute(1, 2, 0))
        plt.axis('off')

        if fpath is None:
            plt.show()
        else:
            plt.savefig(fpath, bbox_inches='tight')

    def to_ndar(self, norm=True, batch=False):
        """Преобразование изображения в массив.

        Args:
            norm (bool): если флаг задан, то будет осуществлена нормализация
                изображения.
            batch (bool): если флаг задан, то возвращает массив размера
                [1, ch, height, width], где ch - это число каналов, height =
                width = size - размер квадратного изображения в пикселях. В
                противном случае возвращает массив размера [ch, height, width].

        """
        tens = self.to_tens(None, norm, batch)
        ndar = tens.numpy()
        return ndar

    def to_tens(self, device=None, norm=True, batch=False):
        """Преобразование изображения в тензор.

        Args:
            device (torch.device): опциональное устройство для переноса объекта.
            norm (bool): если флаг задан, то будет осуществлена нормализация
                изображения.
            batch (bool): если флаг задан, то возвращает тензор размера
                [1, ch, height, width], где ch - это число каналов, height =
                width = size - размер квадратного изображения в пикселях. В
                противном случае возвращает тензор размера [ch, height, width].

        """
        if self.pict is None:
            return

        tens = torchvision.transforms.Compose([
            torchvision.transforms.Resize(self.sz),
            torchvision.transforms.CenterCrop(self.sz),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(self.m, self.v) if norm
                else torchvision.transforms.Lambda(lambda x: x),
            torchvision.transforms.Lambda(lambda x: x[None] if batch else x),
        ])(self.pict)

        return tens if device is None else tens.to(device)
