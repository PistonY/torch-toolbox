__all__ = ['DynamicBatchSampler', 'DynamicSizeImageFolder', 'DistributedDynamicBatchSampler']
import torch
from torch import distributed
from torch.utils.data import BatchSampler, Dataset
from torchvision.datasets import ImageFolder


class DynamicBatchSampler(BatchSampler):
    """DynamicBatchSampler

    Args:
        info_generate_fn (callable): give batch samples extra info.
    """
    def __init__(self, sampler, batch_size: int, drop_last: bool, info_generate_fn=None) -> None:
        super().__init__(sampler, batch_size, drop_last)
        self.info_generate_fn = info_generate_fn if info_generate_fn is not None else lambda: None

    def set_batch_size(self, batch_size: int):
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, " "but got batch_size={}".format(batch_size))
        self.batch_size = batch_size

    def set_info_generate_fn(self, info_generate_fn):
        self.info_generate_fn = info_generate_fn

    def __iter__(self):
        batch = []
        info = self.info_generate_fn()
        for idx in self.sampler:
            batch.append([idx, info])
            if len(batch) == self.batch_size:
                yield batch
                batch = []
                info = self.info_generate_fn()
        if len(batch) > 0 and not self.drop_last:
            yield batch


class DistributedDynamicBatchSampler(BatchSampler):
    """DistributedDynamicBatchSampler to sync all rank info.

    Args:
        info_generate_fn (callable): give batch samples extra info.
        main_rank: rank to send data.
        rank: current rank.

        the result of info_generate_fn must be convert to tensor, current only support integer.
    """
    def __init__(self, sampler, batch_size: int, drop_last: bool, main_rank: int, rank, info_generate_fn=None) -> None:
        super().__init__(sampler, batch_size, drop_last)
        self.info_generate_fn = info_generate_fn if info_generate_fn is not None else lambda: None
        self.main_rank = main_rank
        self.rank = rank
        self.epoch_info = None
        self.reset_and_sync_info()

    def reset_and_sync_info(self):
        epoch_info = [self.info_generate_fn() for _ in range(len(self) + 1)]
        epoch_info = torch.as_tensor(epoch_info, dtype=torch.int, device=torch.device('cuda', self.rank))
        distributed.broadcast(epoch_info, self.main_rank)
        self.epoch_info = epoch_info.cpu().numpy().tolist()

    def set_batch_size(self, batch_size: int):
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, " "but got batch_size={}".format(batch_size))
        self.batch_size = batch_size

    def set_info_generate_fn(self, info_generate_fn):
        self.info_generate_fn = info_generate_fn

    def __iter__(self):
        batch = []
        info = self.epoch_info.pop(0)
        for idx in self.sampler:
            batch.append([idx, info])
            if len(batch) == self.batch_size:
                yield batch
                batch = []
                info = self.epoch_info.pop(0)
        if len(batch) > 0 and not self.drop_last:
            yield batch


class DynamicSizeImageFolder(ImageFolder):
    def __getitem__(self, index_info):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        index, size = index_info
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample, size)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


class DynamicSubset(Dataset):
    r"""
    Subset of a dynamic dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset: Dataset, indices) -> None:
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, index_info):
        index, size = index_info
        index = (self.indices[index], size)
        return self.dataset[index]

    def __len__(self):
        return len(self.indices)
