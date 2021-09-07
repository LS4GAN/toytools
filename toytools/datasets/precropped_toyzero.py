# pylint: disable=missing-module-docstring
import os
import numpy as np

from toytools.collect   import (
    DIR_FAKE, DIR_REAL, load_image, find_images_in_dir, validate_toyzero_images
)
from .generic_dataset   import GenericDataset

class PreCroppedToyzeroDataset(GenericDataset):
    """Toyzero Dataset that loads precropped images.

    Parameters
    ----------
    path : str
        Path where the precropped toyzero dataset is located.
    align_train : bool
        Whenther to align training images.
        Default: False
    align_val : bool
        Whenther to align validation images.
        Default: True
    is_train : bool
        `is_train` flag controls overall behavior of the dataset.
        If `is_train` is True, then the cropped regions of toyzero images are
        selected at random. Otherwise, the cropped regions are selected in a
        reproducible manner using `seed` parameter.  This flag also controls
        which part of the entire toyzero dataset this object will focus on.
        Default: False
    seed : int
        Seed used to initialize random number generators.
        Default: 0
    transform : Callable or None,
        Optional transformation to apply to images.
        E.g. torchvision.transforms.RandomCrop.
        Default: None
    """

    # pylint: disable=too-many-instance-attributes
    # pylint: disable=too-many-arguments
    def __init__(
        self, path,
        align_train      = False,
        align_val        = True,
        is_train         = False,
        seed             = 0,
        transform        = None,
    ):
        super().__init__(path)

        self._is_train  = is_train
        self._path      = path
        self._align     = align_train if is_train else align_val
        self._seed      = seed
        self._transform = transform
        self._prg       = np.random.default_rng(seed)

        self._root        = None
        self._images_fake = None
        self._images_real = None

        self._collect_images()

    def _collect_images(self):

        if self._is_train:
            self._root = os.path.join(self._path, "train")
        else:
            self._root = os.path.join(self._path, "val")

        images_fake = find_images_in_dir(os.path.join(self._root, DIR_FAKE))
        images_real = find_images_in_dir(os.path.join(self._root, DIR_REAL))

        if self._align:
            validate_toyzero_images(images_fake, images_real)
        else:
            self._prg.shuffle(images_fake)
            self._prg.shuffle(images_real)

        self._images_fake = images_fake
        self._images_real = images_real

    def __len__(self):
        return min(len(self._images_fake), len(self._images_real))

    def __getitem__(self, index):
        fname_fake = self._images_fake[index]
        fname_real = self._images_real[index]

        image_fake = load_image(self._root, True,  fname_fake)
        image_real = load_image(self._root, False, fname_real)

        images = [image_fake,  image_real]
        images = [ x.astype(np.float32) for x in images ]

        if self._transform is not None:
            images = [ self._transform(x) for x in images ]

        return images

