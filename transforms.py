#---------------------------------残差图 和 简单二分类---------------------------------
import random
from typing import List, Union
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image_jin, image_yuan, target=None):
        for t in self.transforms:
            image_jin, image_yuan, target = t(image_jin, image_yuan, target)

        return image_jin, image_yuan, target


class ToTensor(object):
    def __call__(self, image_jin, image_yuan, target):
        image_jin = F.to_tensor(image_jin)
        image_yuan = F.to_tensor(image_yuan)
        target = F.to_tensor(target)
        return image_jin, image_yuan, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.flip_prob = prob

    def __call__(self, image_jin, image_yuan, target):
        if random.random() < self.flip_prob:
            image_jin = F.hflip(image_jin)
            image_yuan = F.hflip(image_yuan)
            target = F.hflip(target)
        return image_jin, image_yuan, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image_jin, image_yuan, target):
        image_jin = F.normalize(image_jin, mean=self.mean, std=self.std)
        image_yuan = F.normalize(image_yuan, mean=self.mean, std=self.std)
        return image_jin, image_yuan, target


class Resize(object):
    def __init__(self, size: Union[int, List[int]], resize_mask: bool = True):
        self.size = size  # [h, w]
        self.resize_mask = resize_mask

    def __call__(self, image_jin, image_yuan, target=None):
        image_jin = F.resize(image_jin, self.size)
        image_yuan = F.resize(image_yuan, self.size)
        if self.resize_mask is True:
            target = F.resize(target, self.size)

        return image_jin, image_yuan, target


class RandomCrop(object):
    def __init__(self, size: int):
        self.size = size

    def pad_if_smaller(self, img, fill=0):
        # 如果图像最小边长小于给定size，则用数值fill进行padding
        min_size = min(img.shape[-2:])
        if min_size < self.size:
            ow, oh = img.size
            padh = self.size - oh if oh < self.size else 0
            padw = self.size - ow if ow < self.size else 0
            img = F.pad(img, [0, 0, padw, padh], fill=fill)
        return img

    def __call__(self, image_jin, image_yuan, target):
        image_jin = self.pad_if_smaller(image_jin)
        image_yuan = self.pad_if_smaller(image_yuan)
        crop_params = T.RandomCrop.get_params(image_jin, (self.size, self.size))
        image_jin = F.crop(image_jin, *crop_params)
        image_yuan = F.crop(image_yuan, *crop_params)
        target = F.crop(target, *crop_params)
        return image_jin, image_yuan, target