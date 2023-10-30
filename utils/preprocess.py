from PIL import Image
import random
from torchvision.transforms import ToTensor as torchtotensor
import numpy as np
import random

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


################################iamge################################
class Compose_imglabel(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, label):
        for t in self.transforms:
            img, label = t(img, label)
        return img, label



# class Resize(object):
#     def __init__(self, size):
#         self.size = size
 
#     def __call__(self, image, target=None):
#         image = F.resize(image, self.size, interpolation=F.InterpolationMode.BILINEAR)
#         if target is not None:
#             target = F.resize(target, self.size, interpolation=F.InterpolationMode.NEAREST)
#         return image, target
 
 
class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob
 
    def __call__(self, image, target=None):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            if target is not None:
                target = F.hflip(target)
        return image, target
    

class RandomVerticalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob
 
    def __call__(self, image, target=None):
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            if target is not None:
                target = F.vflip(target)
        return image, target
    

class RandomCrop(object):
    def __init__(self, size):
        self.size = size
 
    def __call__(self, image, target):
        crop_params = T.RandomCrop.get_params(image, self.size)
        image = F.crop(image, *crop_params)
        if target is not None:
            target = F.crop(target, *crop_params)
        return image, target
 
class CenterCrop(object):
    def __init__(self, size):
        self.size = size
 
    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        if target is not None:
            target = F.center_crop(target, self.size)
        return image, target
 
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
 
    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        # if target is not None:
        #     target = F.normalize(target, mean=self.mean, std=self.std)
        return image, target
 
class Pad(object):
    def __init__(self, padding_n, padding_fill_value=0, padding_fill_target_value=0):
        self.padding_n = padding_n
        self.padding_fill_value = padding_fill_value
        self.padding_fill_target_value = padding_fill_target_value
 
    def __call__(self, image, target):
        image = F.pad(image, self.padding_n, self.padding_fill_value)
        if target is not None:
            target = F.pad(target, self.padding_n, self.padding_fill_target_value)
        return image, target
 
class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        if target is not None:
            # target = F.to_tensor(target)
            target = F.to_tensor(target)
            target = torch.gt(target,0.5).long()
        return image, target

    
# class Random_horizontal_flip(object):
#     def _horizontal_flip(self, img, label):
#         # dsa
#         return img.transpose(Image.FLIP_LEFT_RIGHT), label.transpose(Image.FLIP_LEFT_RIGHT)

#     def __init__(self, prob):
#         '''
#         :param prob: should be (0,1)
#         '''
#         assert prob >= 0 and prob <= 1, "prob should be [0,1]"
#         self.prob = prob

#     def __call__(self, img, label):
#         '''
#         flip img and label simultaneously
#         :param img:should be PIL image
#         :param label:should be PIL image
#         :return:
#         '''
#         assert isinstance(img, Image.Image), "should be PIL image"
#         assert isinstance(label, Image.Image), "should be PIL image"
#         if random.random() < self.prob:
#             return self._horizontal_flip(img, label)
#         else:
#             return img, label


class Random_crop_Resize(object):
    def _randomCrop(self, img, label):
        width, height = img.size
        x, y = random.randint(0, self.crop_size), random.randint(0, self.crop_size)
        region = [x, y, width - x, height - y]
        img, label = img.crop(region), label.crop(region)
        img = img.resize((width, height), Image.BILINEAR)
        label = label.resize((width, height), Image.NEAREST)
        return img, label

    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, img, label):
        assert img.size == label.size, "img should have the same shape as label"
        return self._randomCrop(img, label)


class Resize(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, img, label):
        img = img.resize((self.width, self.height), Image.BILINEAR)
        label = label.resize((self.width, self.height), Image.NEAREST)
        return img, label


# class Normalize(object):
#     def __init__(self, mean, std):
#         self.mean, self.std = mean, std

#     def __call__(self, img, label):
#         for i in range(3):
#             img[:, :, i] -= float(self.mean[i])
#         for i in range(3):
#             img[:, :, i] /= float(self.std[i])
#         return img, label


# class toTensor(object):
#     def __init__(self):
#         self.totensor = torchtotensor()

#     def __call__(self, img, label):
#         img, label = self.totensor(img), self.totensor(label).long()
#         return img, label


################################video################################
class Random_crop_Resize_Video(object):
    def _randomCrop(self, img, label, x, y):
        width, height = img.size
        region = [x, y, width - x, height - y]
        img, label = img.crop(region), label.crop(region)
        img = img.resize((width, height), Image.BILINEAR)
        label = label.resize((width, height), Image.NEAREST)
        return img, label

    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, imgs, labels):
        res_img = []
        res_label = []
        x, y = random.randint(0, self.crop_size), random.randint(0, self.crop_size)
        for img, label in zip(imgs, labels):
            img, label = self._randomCrop(img, label, x, y)
            res_img.append(img)
            res_label.append(label)
        return res_img, res_label


class Random_horizontal_flip_video(object):
    def _horizontal_flip(self, img, label):
        return img.transpose(Image.FLIP_LEFT_RIGHT), label.transpose(Image.FLIP_LEFT_RIGHT)

    def __init__(self, prob):
        '''
        :param prob: should be (0,1)
        '''
        assert prob >= 0 and prob <= 1, "prob should be [0,1]"
        self.prob = prob

    def __call__(self, imgs, labels):
        '''
        flip img and label simultaneously
        :param img:should be PIL image
        :param label:should be PIL image
        :return:
        '''
        if random.random() < self.prob:
            res_img = []
            res_label = []
            for img, label in zip(imgs, labels):
                img, label = self._horizontal_flip(img, label)
                res_img.append(img)
                res_label.append(label)
            return res_img, res_label
        else:
            return imgs, labels


class Resize_video(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, imgs, labels):
        res_img = []
        res_label = []
        for img, label in zip(imgs, labels):
            res_img.append(img.resize((self.width, self.height), Image.BILINEAR))
            res_label.append(label.resize((self.width, self.height), Image.NEAREST))
        return res_img, res_label


class Normalize_video(object):
    def __init__(self, mean, std):
        self.mean, self.std = mean, std

    def __call__(self, imgs, labels):
        res_img = []
        for img in imgs:
            for i in range(3):
                img[:, :, i] -= float(self.mean[i])
            for i in range(3):
                img[:, :, i] /= float(self.std[i])
            res_img.append(img)
        return res_img, labels


class toTensor_video(object):
    def __init__(self):
        self.totensor = torchtotensor()

    def __call__(self, imgs, labels):
        res_img = []
        res_label = []
        for img, label in zip(imgs, labels):
            img, label = self.totensor(img), self.totensor(label).long()
            res_img.append(img)
            res_label.append(label)
        return res_img, res_label