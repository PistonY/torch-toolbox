# -*- coding: utf-8 -*-
# @Author  : DevinYang(pistonyang@gmail.com)
import torch


class DataPreFetcher(object):
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        # init
        self.next_data = None
        self.preload()

    def preload(self):

        try:
            self.next_data = next(self.loader)
        except StopIteration:
            self.next_data = None
        with torch.cuda.stream(self.stream):
            if not isinstance(self.next_data, (tuple, list)):
                self.next_data = self.next_data.cuda(non_blocking=True)
            else:
                self.next_data = tuple([d.cuda(non_blocking=True) for d in self.next_data])

    def __next__(self):

        torch.cuda.current_stream().wait_stream(self.stream)
        data = self.next_data

        if data is None:
            raise StopIteration
        self.preload()

        return data

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.loader)
