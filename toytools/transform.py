"""Image transformation tools"""

from typing import Tuple
import numpy as np

def get_background_value(image):
    """Return inferred background value of image"""
    values, counts = np.unique(image, return_counts = True)
    idx_max = np.argmax(counts)
    return values[idx_max]

def get_background_value_fast(image, chunk_size = 128):
    """Return fast but possible incorrect inferred background value of image"""
    return get_background_value(image[:chunk_size, :chunk_size])

def sample_image_region(
    image_shape  : Tuple[int, int],
    region_shape : Tuple[int, int],
    prg          : np.random.Generator
) -> Tuple[int, int, int, int]:
    """Randomly select a subregion of image"""

    x_range = image_shape[0]  - region_shape[0]
    y_range = image_shape[1]  - region_shape[1]

    if (x_range < 0) or (y_range < 0):
        raise RuntimeError(
            "Image shape '%s' is smaller than the region shape '%s'" % (
                image_shape, region_shape
            )
        )

    x0 = prg.integers(0, x_range)
    y0 = prg.integers(0, y_range)

    return (x0, y0, region_shape[0], region_shape[1])

# pylint: disable=missing-function-docstring
def crop_image(image, crop_region):
    if crop_region is None:
        return image

    (x0, y0, width, height) = crop_region
    return image[x0:x0+width, y0:y0+height]

# pylint: disable=missing-function-docstring
def is_image_empty(image, bkg_value, min_signal):
    n_signal_pixels = np.count_nonzero(image != bkg_value)

    return (n_signal_pixels < min_signal)

# pylint: disable=too-many-arguments
def try_find_region_with_signal(
    image        : np.ndarray,
    prg          : np.random.Generator,
    bkg_value    : int,
    min_signal   : int,
    region_shape : Tuple[int, int],
    retries      : int,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Try find region of image that contains signal"""

    crop_region   = None
    cropped_image = None

    for _ in range(max(1, retries)):
        crop_region   = sample_image_region(image.shape, region_shape, prg)
        cropped_image = crop_image(image, crop_region)

        if not is_image_empty(cropped_image, bkg_value, min_signal):
            break

    return (cropped_image, crop_region)

