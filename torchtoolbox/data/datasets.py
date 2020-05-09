# -*- coding: utf-8 -*-
# @Author  : DevinYang(pistonyang@gmail.com)
__all__ = ['NonLabelDataset', 'FeaturePairDataset', 'FeaturePairBin']

import glob
import os
import pickle
import numpy as np
from torch.utils.data import Dataset
from .utils import decode_img_from_buf, cv_loader


class NonLabelDataset(Dataset):
    """This is used for label-free training like GAN, VAE...

    root/xxx.jpg
    root/xxy.jpg
    root/xxz.jpg

    Args:
        root_dir (str): root dir of data.
        transform (callable): transform func.
    """

    def __init__(self, root_dir, transform=None, loader=cv_loader):
        self.transform = transform
        self.items = sorted(os.listdir(root_dir))
        self.items = [os.path.join(root_dir, f) for f in self.items]
        self.loader = loader

    def __len__(self):
        return len(self.items)

    def __getitem__(self, item):
        img = self.loader(self.items[item])
        if self.transform:
            img = self.transform(img)
        return img


class FeaturePairDataset(Dataset):
    """File structure should be like this.

    root/
        xxx/
            aaa.jpg
            bbb.jpg
        yyy/
            ...
        zzz/
            ...
        is_same(.txt)

    is_same file structure should be like this.

    is_same.txt
        xxx 1
        yyy 0
        zzz 0
    """

    def __init__(
            self,
            root,
            is_same_file=None,
            transform=None,
            loader=cv_loader):
        self.root = root
        is_same = os.path.join(
            root, 'is_same.txt' if is_same_file is None else is_same_file)
        is_same_list = []
        with open(is_same) as f:
            for line in f.readlines():
                is_same_list.append(line.replace('\n', '').split(' '))
        self.file_list = is_same_list
        self.transform = transform
        self.loader = loader
        self.pre_check()

    def pre_check(self):
        self.file_list = [
            [
                glob.glob(
                    os.path.join(
                        self.root,
                        dir_name,
                        '*.jpg')),
                int(is_same)] for dir_name,
            is_same in self.file_list]
        for files, _ in self.file_list:
            assert len(files) == 2

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        pair, is_same = self.file_list[item]
        imgs = list(map(self.loader, pair))
        if self.transform:
            imgs = list(map(self.transform, imgs))
        return imgs, is_same


class FeaturePairBin(Dataset):
    """A dataset wrapping over a pickle serialized (.bin) file provided by InsightFace Repo.

    Parameters
    ----------
    name : str. Name of val dataset.
    root : str. Path to face folder.
    transform : callable, default None
        A function that takes data and transforms them.

    """

    def __init__(self, name, root, transform=None, backend='cv2'):
        self._transform = transform
        self.name = name
        with open(os.path.join(root, "{}.bin".format(name)), 'rb') as f:
            self.bins, self.issame_list = pickle.load(f, encoding='iso-8859-1')

        self._do_encode = not isinstance(self.bins[0], np.ndarray)
        self.backend = backend

    def _decode(self, im):
        if self._do_encode:
            im = im.encode("iso-8859-1")
        im = decode_img_from_buf(im, self.backend)
        return im

    def __getitem__(self, idx):
        img0 = self._decode(self.bins[2 * idx])
        img1 = self._decode(self.bins[2 * idx + 1])

        issame = 1 if self.issame_list[idx] else 0

        if self._transform is not None:
            img0 = self._transform(img0)
            img1 = self._transform(img1)

        return (img0, img1), issame

    def __len__(self):
        return len(self.issame_list)
