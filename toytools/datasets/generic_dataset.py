# pylint: disable=missing-module-docstring

class GenericDataset:
    """Abstract class for toyzero Dataset"""

    def __init__(self, path):
        self._path = path

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

