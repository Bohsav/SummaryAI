import torch
from torch.utils.data import Dataset, IterableDataset
from load_data import Loader


# TODO: Design custom map dataset with __len__ and __getitem__(id)
class CustomMapDataset(Dataset):
    def __init__(self):
        raise NotImplementedError("")

    def __len__(self):
        raise NotImplementedError("")

    def __getitem__(self, index):
        raise NotImplementedError("")


# TODO: Design the iterable dataset wrapper
class CustomIterableDataset(IterableDataset):
    def __init__(self):
        raise NotImplementedError("")

    def __iter__(self):
        raise NotImplementedError("")

    def __next__(self):
        raise NotImplementedError("")