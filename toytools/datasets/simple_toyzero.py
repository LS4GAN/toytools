# pylint: disable=missing-module-docstring
import numpy as np

from toytools.collect   import (
    collect_toyzero_images, filter_images, load_image, train_test_split
)
from toytools.transform import (
    get_background_value_fast, crop_image, try_find_region_with_signal
)
from .generic_dataset import GenericDataset

CROP_RETRIES   = 100
MIN_PIXEL_DIFF = 10

# pylint: disable=too-many-instance-attributes
class SimpleToyzeroDataset(GenericDataset):
    """Simple Dataset to load toyzero images on demand with no preprocessing.

    Parameters
    ----------
    path : str
        Path where the toyzero dataset is located.
    crop_shape : (int, int) or None
        If specified, then crop images down to shape specified by `crop_shape`.
        Default: None
    expansion_factor : int
        Number of cropped regions to select from a single image.
        Default: 1
    is_train : bool
        `is_train` flag controls overall behavior of the
        `SimpleToyzeroDataset`. If `is_train` is True, then the cropped regions
        of toyzero images are selected at random. Otherwise, the cropped
        regions are selected in a reproducible manner using `seed` parameter.
        This flag also controls which part of the entire toyzero dataset this
        object will focus on.
        Default: False
    planes : Set[str] or None
        A set of planes to select. If None, then all planes are selected.
        Default: None
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
        self,
        path,
        crop_shape       = None,
        expansion_factor = 1,
        is_train         = False,
        planes           = None,
        seed             = 0,
        transform        = None,
        val_size         = 0.2,
    ):
        super().__init__(path)

        self._is_train = is_train
        self._images   = []

        self._indices   = []
        self._len       = 0

        self._crop_shape = crop_shape
        self._transform  = transform
        self._seed       = seed
        self._efactor    = expansion_factor
        self._prg        = np.random.default_rng(seed)
        self._planes     = planes
        self._val_size   = val_size

        self._collect_images()
        self._split_dataset()

    def _collect_images(self):
        """Locate all matching images in the dataset directory."""
        images = collect_toyzero_images(self._path)
        self._images = filter_images(images, None, self._planes)

    def _split_dataset(self):
        """Split images into training/validation samples."""
        val_size = self._val_size

        if val_size > 1:
            # Temporary reduce size
            val_size = max(1, val_size // self._efactor)

        train_indices, val_indices = train_test_split(
            len(self._images), val_size, shuffle = True, prg = self._prg
        )

        if self._is_train:
            self._indices = train_indices
        else:
            self._indices = val_indices

        self._len = len(self._indices) * self._efactor

    def __len__(self):
        return self._len

    def _get_rand_cropped_area(self, image_fake, image_real, prg, bkg_value):
        """Randomly crop images."""
        cropped_fake_image, crop_region = try_find_region_with_signal(
            image_fake, prg, bkg_value, MIN_PIXEL_DIFF, self._crop_shape,
            CROP_RETRIES
        )

        return (cropped_fake_image, crop_image(image_real, crop_region))

    def __getitem__(self, index):

        if self._is_train:
            prg          = np.random.default_rng()
            sample_index = prg.choice(self._indices, size = 1)[0]
        else:
            prg          = np.random.default_rng(index)
            sample_index = self._indices[index % len(self._indices)]

        image_name = self._images[sample_index]
        image_fake = load_image(self._path, True,  image_name)
        image_real = load_image(self._path, False, image_name)
        bkg_value  = get_background_value_fast(image_fake)

        if self._crop_shape is not None:
            (image_fake, image_real) = self._get_rand_cropped_area(
                image_fake, image_real, prg, bkg_value
            )

        images = [ (x - bkg_value) for x in (image_fake, image_real) ]
        images = [ x.astype(np.float32) for x in images ]

        if self._transform is not None:
            images = [ self._transform(x)   for x in images ]

        return images

