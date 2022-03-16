# pylint: disable=missing-module-docstring
import os
import numpy as np

from toytools.collect import find_images_in_dir
from toytools.consts  import SPLIT_TRAIN, SPLIT_TEST, SPLIT_VAL
from .generic_dataset import GenericDataset, DOMAIN_MAP

class PreCroppedToyzeroDatasetV1(GenericDataset):
    """Toyzero Dataset that loads precropped images.

    Parameters
    ----------
    path : str
        Path where the precropped toyzero dataset is located.
    domain : str
        Choices: 'a' ('real'), 'b' ('fake')
    split : str
        Choices: 'train', 'test', 'val'
    transform : Callable or None,
        Optional transformation to apply to images.
        E.g. torchvision.transforms.RandomCrop.
        Default: None
    """

    def __init__(self, path, domain, split, transform = None):
        super().__init__(path)

        assert domain in DOMAIN_MAP
        assert split  in [ SPLIT_TRAIN, SPLIT_TEST, SPLIT_VAL ]

        self._root      = os.path.join(path, split, DOMAIN_MAP[domain])
        self._domain    = domain
        self._split     = split
        self._transform = transform
        self._images    = find_images_in_dir(self._root)

    def __len__(self):
        return len(self._images)

    def __getitem__(self, index):
        fname = self._images[index]

        with np.load(os.path.join(self._root, fname)) as f:
            image = f[f.files[0]].astype(np.float32)

        if self._transform is not None:
            image = self._transform(image)

        return image

