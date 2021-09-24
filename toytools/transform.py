"""Image transformation tools"""

from typing import Tuple
import numpy as np
import torch
import torch.nn as nn

def get_background_value(image):
    """Return inferred background value of image"""
    values, counts = np.unique(image, return_counts = True)
    idx_max = np.argmax(counts)
    return values[idx_max]

def get_background_value_fast(image, chunk_size = 128):
    """Return fast but possibly incorrect inferred background value of image"""
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


#===================================================================================
# ============================= Find Multi-track windows ===========================
def multitrack_preprocess(x, kernel_size=2, stride=2):
    """
    Preprocessing steps for multitrack detections.
    Input:
        - x (numpy.ndarray): the input image of shape (H, W)
        - kernel_size (int): the kernel size of torch.AvgPool2d
        - stride (int): the stride of torch.AvgPool2d
    Output (numpy.ndarray): the processed 0-1 image
    """
    x = np.abs(x)
    x[x > 0] = 1
    with torch.no_grad():
        x = torch.as_tensor(x).to(torch.float32)
        input_tensor = x.view(1, 1, x.shape[0], x.shape[1])
        avgpool = nn.AvgPool2d(kernel_size, stride=stride)
        output = avgpool(input_tensor)
        output[output > 0] = 1
    return output.squeeze().numpy()


def scan_simple(vec):
    """
    Given a 0-1 vector, determine the number of consecutive sections of 1s.
    For example, if the vector is [0,0,1,1,1,0,1], scan should return 2.
    Input:
        - vec (any 1-dimensional iterable): a 0-1 vector
    Output (bool): whether vec contains at least two consecutive sections of 1s
    """
    return np.count_nonzero(np.abs(vec[1:] - vec[:-1])) >= 3

def isMultiTrack(x, kernel_size=2, stride=2, fraction=.05):
    """
    Determined the presence of multitracks
    Input:
        - x (numpy.ndarray): the input image of x shape (H, W)
        - kernel_size (int): the kernel size of torch.AvgPool2d
        - stride (int): the stride of torch.AvgPool2d
        - fraction (float): the fraction of 1-d scan vectors
            with more than one consecutive sections of 1s to
            qualify a image as having multi-tracks.
    Output (bool): whether the image x has multiple tracks.
    """
    y = multitrack_preprocess(x, kernel_size=kernel_size, stride=stride)
    # scan vertically
    v = np.array([scan_simple(row) for row in y])
    # scan horizontally
    h = np.array([scan_simple(y[:, j]) for j in range(len(y[0]))])
    m = sum(v) + sum(h)
    return m / (len(v) + len(h)) >= fraction
# ============================= Find Multi-track windows ===========================
#===================================================================================
