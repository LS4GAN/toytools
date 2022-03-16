# pylint: disable=missing-module-docstring
import os
import numpy as np
import pandas as pd

from toytools.collect   import train_val_test_split
from toytools.transform import crop_image
from toytools.consts    import SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST

from .generic_dataset   import GenericDataset, DOMAIN_MAP

class PreSimpleToyzeroDatasetV1(GenericDataset):
    """Toyzero Dataset that loads images according to preprocessed list.

    Parameters
    ----------
    path : str
        Path where the toyzero dataset is located.
    fname : str
        Name of the file containing preprocessed list of toyzero images.
        This should be located under `path`.
    domain : str
        Choices: 'a' ('real'), 'b' ('fake')
    split : str
        Choices: 'train', 'test', 'val'
    shuffle : bool
        Controls whether to shuffle the dataset before splitting it into
        training/validation/test parts.
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
    test_size : float or int
        Fraction of the toyzero dataset that will be used as a test
        sample. If test_size <= 1, then it is treated as a fraction, i.e.
        (size of val sample) = `test_size` * (size of toyzero dataset)
        Otherwise, (size of val sample) = `test_size`.
        Default: 0.2
    """

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments
    def __init__(
        self, path,
        fname            = None,
        split            = SPLIT_TRAIN,
        domain           = 'a',
        seed             = 0,
        shuffle          = True,
        transform        = None,
        val_size         = 0.2,
        test_size        = 0,
    ):
        super().__init__(path)

        assert split  in [ SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST ]
        assert domain in DOMAIN_MAP

        self._domain    = domain
        self._split     = split
        self._seed      = seed
        self._transform = transform
        self._prg       = np.random.default_rng(seed)
        self._val_size  = val_size
        self._test_size = test_size
        self._df        = None
        self._root      = os.path.join(path, DOMAIN_MAP[domain])

        df = pd.read_csv(os.path.join(path, fname), index_col = 'index')
        self._split_dataset(df, shuffle)

    def _split_dataset(self, df, shuffle):
        """Split dataset into training/validation/test parts."""
        train_indices, val_indices, test_indices = train_val_test_split(
            len(df), self._val_size, self._test_size, shuffle, self._prg
        )

        if self._split == SPLIT_TRAIN:
            indices = train_indices

        elif self._split == SPLIT_VAL:
            indices = val_indices

        else:
            indices = test_indices

        self._df = df.iloc[indices]

    def __len__(self):
        return len(self._df)

    def __getitem__(self, index):
        sample = self._df.iloc[index]

        with np.load(os.path.join(self._root, sample.image)) as f:
            image = f[f.files[0]]

        image = crop_image(
            image, (sample.x, sample.y, sample.width, sample.height)
        )

        image = image - sample.bkg
        image = image.astype(np.float32)

        if self._transform is not None:
            image = self._transform(image)

        return image

