# pylint: disable=missing-module-docstring
import os
import numpy as np
import pandas as pd

from toytools.collect   import load_image
from toytools.transform import crop_image
from .generic_dataset   import GenericDataset

COLUMNS = [
    'image', 'event', 'apa', 'plane', 'x', 'y', 'width', 'height', 'bkg'
]

class PreSimpleToyzeroDataset(GenericDataset):
    """Toyzero Dataset that loads images according to preprocessed list.

    Parameters
    ----------
    path : str
        Path where the toyzero dataset is located.
    fname : str
        Name of the file containing preprocessed list of toyzero images.
        This should be located under `path`.
    is_train : bool
        `is_train` flag controls overall behavior of the
        `SimpleToyzeroDataset`. If `is_train` is True, then the cropped regions
        of toyzero images are selected at random. Otherwise, the cropped
        regions are selected in a repdorucible manner using `seed` parameter.
        This flag also controls which part of the entire toyzero dataset this
        object will focus on.
        Default: False
    shuffle : bool
        Controls whether to shuffle the dataset before splitting it into
        training and validation parts.
        Default: True
    seed : int
        Seed used to initialize random number generators.
        Default: 0
    transform : Callable or None,
        Optional transformation to apply to images.
        E.g. torchvision.transforms.RandomCrop.
        Default: None
    val_size : float or int
        Fraction of the toyzero dataset that will be used as a validation
        sample. If val_size <= 1, then it is treated as a fraction, i.e.
        (size of val sample) = `val_size` * (size of toyzero dataset)
        Otherwise, (size of val sample) = `val_size`.
        Default: 0.2
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self, path,
        fname            = None,
        is_train         = False,
        seed             = 0,
        shuffle          = True,
        transform        = None,
        val_size         = 0.2,
    ):
        super().__init__(path)

        self._is_train  = is_train
        self._seed      = seed
        self._shuffle   = shuffle
        self._transform = transform
        self._prg       = np.random.default_rng(seed)
        self._val_size  = val_size

        self._df = pd.read_csv(os.path.join(path, fname), index_col = 'index')
        self._split_dataset()

    def _split_dataset(self):
        """Split dataset into training/validation parts."""
        indices = np.arange(len(self._df))

        if self._shuffle:
            self._prg.shuffle(indices)

        val_size = self._val_size

        if val_size <= 1:
            val_size = int(len(indices) * val_size)

        train_size = len(indices) - val_size

        if self._is_train:
            indices = indices[:train_size]
        else:
            indices = indices[train_size:]

        self._df = self._df.iloc[indices]

    def __len__(self):
        return len(self._df)

    def __getitem__(self, index):
        sample = self._df.iloc[index]

        image_fake = load_image(self._path, True,  sample.image)
        image_real = load_image(self._path, False, sample.image)

        crop_region = (sample.x, sample.y, sample.width, sample.height)

        images = [ image_fake, image_real ]

        images = [ crop_image(x, crop_region) for x in images ]
        images = [ (x - sample.bkg)           for x in images ]
        images = [ x.astype(np.float32)       for x in images ]

        if self._transform is not None:
            images = [ self._transform(x) for x in images ]

        return images

