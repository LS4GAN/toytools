# pylint: disable=missing-module-docstring

from toytools.collect import DIR_FAKE, DIR_REAL

DOMAIN_MAP = {
    'a'    : DIR_FAKE,
    'fake' : DIR_FAKE,
    'b'    : DIR_REAL,
    'real' : DIR_REAL,
}

class GenericDataset:
    """Abstract class for toyzero Dataset"""

    def __init__(self, path):
        self._path = path

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

