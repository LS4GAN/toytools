"""Functions to construct pytorch datasets from toytools datasets"""

from torch.utils.data import Dataset
from .funcs import get_toyzero_dataset

class TorchDatasetWrapper(Dataset):
    """Wrapper around GenericDataset that creates torch.Dataset"""

    def __init__(self, dset, **kwargs):
        super().__init__(**kwargs)
        self._dset = dset

    def __getitem__(self, index):
        return self._dset[index]

    def __len__(self):
        return len(self._dset)

def get_toyzero_dataset_torch(name, path, **data_args):
    """Return pytorch toyzero dataset based on its name"""
    generic_dataset = get_toyzero_dataset(name, path, **data_args)
    return TorchDatasetWrapper(generic_dataset)

