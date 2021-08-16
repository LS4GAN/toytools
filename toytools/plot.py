"""A collection of functions to simplify plotting"""

from typing import Union, List

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def drop_ticks(ax):
    """Remove axes ticks and labels"""

    ax.tick_params(
        axis = 'both', which = 'both',
        bottom = False, top = False, left = False, right = False,
        labelbottom = False, labelleft = False
    )

def get_common_images_range(images, margin = 0.3):
    """Get common color range across multiple images"""

    vmax = max([ np.max(image) for image in images ])
    vmax = max(vmax, 1)
    vmax = (1 + margin) * vmax

    vmin = min([ np.min(image) for image in images ])
    vmin = min(vmin, -1)
    vmin = (1 + margin) * vmin

    return (vmin, vmax)

def default_image_plot(ax, image, vmin = None, vmax = None):
    """Plot toyzero `image` with default style."""

    drop_ticks(ax)

    if vmin is None:
        vmin = min(np.min(image), -1)

    if vmax is None:
        vmax = max(np.max(image), 1)

    # pylint: disable=no-member
    cmap = mpl.cm.seismic
    norm = mpl.colors.TwoSlopeNorm(0, vmin, vmax)

    return ax.imshow(image, cmap = cmap, norm = norm)

def save_figure(f : plt.Figure, path : str, ext : Union[str, List[str]]):
    """Save pyplot figure in various formats, specified by `ext`."""

    if isinstance(ext, list):
        for e in ext:
            save_figure(f, path, e)
        return

    f.savefig(path + '.' + ext, bbox_inches = 'tight')

