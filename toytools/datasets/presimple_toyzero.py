# pylint: disable=missing-module-docstring
import os
import ctypes
import multiprocessing as mp
import numpy as np
import pandas as pd

from toytools.collect   import load_image, train_test_split
from toytools.transform import crop_image
from .generic_dataset   import GenericDataset

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
        regions are selected in a reproducible manner using `seed` parameter.
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
        is_in_mem        = False,
    ):
        super().__init__(path)

        self._is_train  = is_train
        self._seed      = seed
        self._shuffle   = shuffle
        self._transform = transform
        self._prg       = np.random.default_rng(seed)
        self._val_size  = val_size
        self._is_in_mem = is_in_mem

        self._df = pd.read_csv(os.path.join(path, fname), index_col = 'index')
        self._df = self._split_dataset()
        if is_in_mem:
            self._shared_data = self._preload_data()

    def _split_dataset(self):
        """Split dataset into training/validation parts."""
        train_indices, val_indices = train_test_split(
            len(self._df), self._val_size, self._shuffle, self._prg
        )

        if self._is_train:
            indices = train_indices
        else:
            indices = val_indices

        return self._df.iloc[indices]

    def _preload_data(self):
        data_sz = len(self._df)
        img, _ = self._load_image_pair(0)
        h, w = img.shape
        shared_alloc = mp.Array(ctypes.c_float, data_sz * 2 * h * w)
        shared_data = np.ctypeslib.as_array(shared_alloc.get_obj())
        shared_data = shared_data.reshape(data_sz, 2, h, w)
        for i in range(data_sz):
            shared_data[i][0], shared_data[i][1] = self._load_image_pair(i)
        return shared_data

    def _load_image_pair(self, index):
        sample = self._df.iloc[index]

        image_fake = load_image(self._path, True,  sample.image)
        image_real = load_image(self._path, False, sample.image)

        crop_region = (sample.x, sample.y, sample.width, sample.height)

        images = [ image_fake, image_real ]

        images = [ crop_image(x, crop_region) for x in images ]
        images = [ (x - sample.bkg)           for x in images ]
        images = [ x.astype(np.float32)       for x in images ]

        return images

    def __len__(self):
        return len(self._df)

    def __getitem__(self, index):
        if self._is_in_mem:
            images = [self._shared_data[index][0], self._shared_data[index][1]]
        else:
            images = self._load_image_pair(index)

        if self._transform is not None:
            images = [ self._transform(x) for x in images ]

        return images
