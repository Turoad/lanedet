import random
import cv2
import numpy as np
import torch
import numbers
import collections
from PIL import Image

from ..registry import PROCESS

def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')


@PROCESS.register_module
class ToTensor(object):
    """Convert some results to :obj:`torch.Tensor` by given keys.

    Args:
        keys (Sequence[str]): Keys that need to be converted to Tensor.
    """

    def __init__(self, keys=['img', 'mask'], cfg=None):
        self.keys = keys

    def __call__(self, sample):
        data = {}
        if len(sample['img'].shape) < 3:
            sample['img'] = np.expand_dims(sample['img'], -1)
        for key in self.keys:
            data[key] = to_tensor(sample[key])
        data['img'] = data['img'].permute(2, 0, 1)
        return data

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'


@PROCESS.register_module
class RandomLROffsetLABEL(object):
    def __init__(self,max_offset, cfg=None):
        self.max_offset = max_offset
    def __call__(self, sample):
        img = sample['img'] 
        label = sample['mask'] 
        offset = np.random.randint(-self.max_offset,self.max_offset)
        h, w = img.shape[:2]

        img = np.array(img)
        if offset > 0:
            img[:,offset:,:] = img[:,0:w-offset,:]
            img[:,:offset,:] = 0
        if offset < 0:
            real_offset = -offset
            img[:,0:w-real_offset,:] = img[:,real_offset:,:]
            img[:,w-real_offset:,:] = 0

        label = np.array(label)
        if offset > 0:
            label[:,offset:] = label[:,0:w-offset]
            label[:,:offset] = 0
        if offset < 0:
            offset = -offset
            label[:,0:w-offset] = label[:,offset:]
            label[:,w-offset:] = 0
        sample['img'] = img
        sample['mask'] = label
        
        return sample 

@PROCESS.register_module
class RandomUDoffsetLABEL(object):
    def __init__(self,max_offset, cfg=None):
        self.max_offset = max_offset
    def __call__(self, sample):
        img = sample['img'] 
        label = sample['mask'] 
        offset = np.random.randint(-self.max_offset,self.max_offset)
        h, w = img.shape[:2]

        img = np.array(img)
        if offset > 0:
            img[offset:,:,:] = img[0:h-offset,:,:]
            img[:offset,:,:] = 0
        if offset < 0:
            real_offset = -offset
            img[0:h-real_offset,:,:] = img[real_offset:,:,:]
            img[h-real_offset:,:,:] = 0

        label = np.array(label)
        if offset > 0:
            label[offset:,:] = label[0:h-offset,:]
            label[:offset,:] = 0
        if offset < 0:
            offset = -offset
            label[0:h-offset,:] = label[offset:,:]
            label[h-offset:,:] = 0
        sample['img'] = img
        sample['mask'] = label
        return sample 

@PROCESS.register_module
class Resize(object):
    def __init__(self, size, cfg=None):
        assert (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size

    def __call__(self, sample):
        out = list()
        sample['img'] = cv2.resize(sample['img'], self.size,
                              interpolation=cv2.INTER_CUBIC)
        if 'mask' in sample:
            sample['mask'] = cv2.resize(sample['mask'], self.size,
                                  interpolation=cv2.INTER_NEAREST)
        return sample


@PROCESS.register_module
class RandomCrop(object):
    def __init__(self, size, cfg=None):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img_group):
        h, w = img_group[0].shape[0:2]
        th, tw = self.size

        out_images = list()
        h1 = random.randint(0, max(0, h - th))
        w1 = random.randint(0, max(0, w - tw))
        h2 = min(h1 + th, h)
        w2 = min(w1 + tw, w)

        for img in img_group:
            assert (img.shape[0] == h and img.shape[1] == w)
            out_images.append(img[h1:h2, w1:w2, ...])
        return out_images


@PROCESS.register_module
class CenterCrop(object):
    def __init__(self, size, cfg=None):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img_group):
        h, w = img_group[0].shape[0:2]
        th, tw = self.size

        out_images = list()
        h1 = max(0, int((h - th) / 2))
        w1 = max(0, int((w - tw) / 2))
        h2 = min(h1 + th, h)
        w2 = min(w1 + tw, w)

        for img in img_group:
            assert (img.shape[0] == h and img.shape[1] == w)
            out_images.append(img[h1:h2, w1:w2, ...])
        return out_images

@PROCESS.register_module
class RandomRotation(object):
    def __init__(self, degree=(-10, 10), interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST), padding=None, cfg=None):
        self.degree = degree
        self.interpolation = interpolation
        self.padding = padding
        if self.padding is None:
            self.padding = [0, 0]

    def _rotate_img(self, sample, map_matrix):
        h, w = sample['img'].shape[0:2]
        sample['img'] = cv2.warpAffine(
            sample['img'], map_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=self.padding)

    def _rotate_mask(self, sample, map_matrix):
        if 'mask' not in sample:
            return
        h, w = sample['mask'].shape[0:2]
        sample['mask'] = cv2.warpAffine(
            sample['mask'], map_matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=self.padding)


    def __call__(self, sample):
        v = random.random()
        if v < 0.5:
            degree = random.uniform(self.degree[0], self.degree[1])
            h, w = sample['img'].shape[0:2]
            center = (w / 2, h / 2)
            map_matrix = cv2.getRotationMatrix2D(center, degree, 1.0)
            self._rotate_img(sample, map_matrix)
            self._rotate_mask(sample, map_matrix)
        return sample


@PROCESS.register_module
class RandomBlur(object):
    def __init__(self, applied, cfg=None):
        self.applied = applied

    def __call__(self, img_group):
        assert (len(self.applied) == len(img_group))
        v = random.random()
        if v < 0.5:
            out_images = []
            for img, a in zip(img_group, self.applied):
                if a:
                    img = cv2.GaussianBlur(
                        img, (5, 5), random.uniform(1e-6, 0.6))
                out_images.append(img)
                if len(img.shape) > len(out_images[-1].shape):
                    out_images[-1] = out_images[-1][...,
                                                    np.newaxis]  # single channel image
            return out_images
        else:
            return img_group


@PROCESS.register_module
class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given numpy Image with a probability of 0.5
    """

    def __init__(self, cfg=None):
        pass

    def __call__(self, sample):
        v = random.random()
        if v < 0.5:
            sample['img'] = np.fliplr(sample['img'])
            if 'mask' in sample: sample['mask'] = np.fliplr(sample['mask'])
        return sample


@PROCESS.register_module
class Normalize(object):
    def __init__(self, img_norm, cfg=None):
        self.mean = np.array(img_norm['mean'], dtype=np.float32)
        self.std = np.array(img_norm['std'], dtype=np.float32)

    def __call__(self, sample):
        m = self.mean
        s = self.std
        img = sample['img'] 
        if len(m) == 1:
            img = img - np.array(m)  # single channel image
            img = img / np.array(s)
        else:
            img = img - np.array(m)[np.newaxis, np.newaxis, ...]
            img = img / np.array(s)[np.newaxis, np.newaxis, ...]
        sample['img'] = img

        return sample 
