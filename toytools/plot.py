"""A collection of functions to simplify plotting"""

from typing import Union, List

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def decorate_axes(
    ax, ticks = False, grid = False, minor = False, label = False
):
    """Modify axes style"""
    ax.tick_params(
        axis = 'both', which = 'both',
        bottom = ticks, top = ticks, left = ticks, right = ticks,
    )

    if not label:
        ax.tick_params(
            axis = 'both', which = 'both',
            labelbottom = False, labelleft = False
        )

    if minor and ticks:
        ax.minorticks_on()

    if grid:
        ax.grid(True, which = 'major', linestyle = 'dashed', linewidth = 1.0)

        if minor:
            ax.grid(
                True, which = 'minor', linestyle = 'dashed', linewidth = 0.5
            )

def get_common_images_range(images, margin = 0.3, symmetric = True):
    """Get common color range across multiple images"""

    vmax = max([ np.max(image) for image in images ])
    vmax = (1 + margin) * vmax

    vmin = min([ np.min(image) for image in images ])
    vmin = (1 + margin) * vmin

    if symmetric:
        vmax = max(vmax, -vmin)
        vmin = -vmax

    return (vmin, vmax)

# pylint: disable=too-many-arguments
def default_image_plot(
    ax, image, vrange = None, ticks = False, grid = False, symmetric = True,
    symlog = False
):
    """Plot toyzero `image` with default style."""
    decorate_axes(ax, ticks, grid, minor = True, label = False)

    if vrange is None:
        vrange = get_common_images_range([ image, ], symmetric = symmetric)

    # pylint: disable=no-member
    cmap = mpl.cm.seismic

    if symlog:
        norm = mpl.colors.SymLogNorm(
            linthresh = 1.0, vmin = vrange[0], vmax = vrange[1]
        )
    else:
        norm = mpl.colors.TwoSlopeNorm(0, vrange[0], vrange[1])

    return ax.imshow(image, cmap = cmap, norm = norm)

def save_figure(f : plt.Figure, path : str, ext : Union[str, List[str]]):
    """Save pyplot figure in various formats, specified by `ext`."""

    if isinstance(ext, list):
        for e in ext:
            save_figure(f, path, e)
        return

    f.savefig(path + '.' + ext, bbox_inches = 'tight')

