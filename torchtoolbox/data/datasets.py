# -*- coding: utf-8 -*-
# @Author  : DevinYang(pistonyang@gmail.com)
__all__ = ['NonLabelDataset']
import os
from PIL import Image
from torch.utils.data import Dataset


class NonLabelDataset(Dataset):
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
