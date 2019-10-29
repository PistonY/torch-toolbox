# -*- coding: utf-8 -*-
# @Author  : DevinYang(pistonyang@gmail.com)
__all__ = ['NonLabelDataset']
import glob
import os
from PIL import Image
from torch.utils.data import Dataset


class NonLabelDataset(Dataset):
    """This is used for label-free training like GAN, VAE...

    root/xxx.jpg
    root/xxy.jpg
    root/xxz.jpg

    Args:
        root_dir (str): root dir of data.
        transform (callable): transform func.
    """

    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.items = os.listdir(root_dir)
        self.items = [os.path.join(root_dir, f) for f in self.items]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, item):
        img = Image.open(self.items[item]).convert('RGB')
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

    def __init__(self, root, is_same_file=None, transform=None):
        self.root = root
        is_same = os.path.join(root, 'is_same.txt' if is_same_file is None else is_same_file)
        is_same_list = []
        with open(is_same) as f:
            for line in f.readlines():
                is_same_list.append(line.replace('\n', '').split(' '))
        self.file_list = is_same_list
        self.transform = transform

    def __getitem__(self, item):
        dir = os.path.join(self.root, self.file_list[item][0])
        is_same = int(self.file_list[item][1])
        imgs = glob.glob(os.path.join(dir, '*.jpg'))
        assert len(imgs) == 2
        img0, img1 = map(Image.open, (imgs[0], imgs[1]))
        if self.transform:
            img0, img1 = map(self.transform, (img0, img1))
        return (img0, img1), is_same

    def __len__(self):
        return len(self.file_list)
