"""Definitions of custom colormaps"""

import numpy as np
import matplotlib as mpl

RANGE_LOW = (0.45, 0.55)
RANGE_MID = (0.4, 0.6)

class ToyzeroNorm(mpl.colors.Normalize):
    """Normalization for the toyzero colormap.

    This norm maps values:
      - from (0, threshold_mid)                 to (0.5, RANGE_LOW[1])
      - from (threshold_mid, threshold_high)    to (RANGE_LOW[1], RANGE_MID[1])
      - from (threshold_high, maxval)           to (RANGE_MID[1], 1)
    and symmetrically for the negative values.
    """

    def __init__(
        self, vrange, threshold_mid = 5, threshold_high = 10, margin = 0.1
    ):
        self.maxval  = max(vrange[1], -vrange[0], 2 * threshold_high)
        self.maxval *= (1 + margin)
        super().__init__(vmin = -self.maxval, vmax = self.maxval, clip = False)

        self._th_mid  = threshold_mid
        self._th_high = threshold_high

        self._range_from, self._range_to = self._get_interval_map()

    def _get_interval_map(self):
        xy = [
            (-self.maxval,   0),
            (-self._th_high, RANGE_MID[0]),
            (-self._th_mid,  RANGE_LOW[0]),
            (0,              0.5),
            (self._th_mid,   RANGE_LOW[1]),
            (self._th_high,  RANGE_MID[1]),
            (self.maxval,    1),
        ]

        x, y = list(zip(*xy))

        return (x, y)

    def __call__(self, value, clip = None):
        value, _ = mpl.colors.Normalize.process_value(value)
        return np.interp(
            value, xp = self._range_from, fp = self._range_to,
        )

    def inverse(self, value):
        value, _ = mpl.colors.Normalize.process_value(value)
        return np.interp(
            value, xp = self._range_to, fp = self._range_from,
        )

def get_toyzero_cmap(
    cmap_low = 'RdGy_r', cmap_mid = 'viridis', cmap_high = 'seismic',
    washout_low = 1.5, washout_mid = 1.2, N_low = 10, N_mid = 20, N_high = 257
):
    # pylint: disable=too-many-arguments
    """Construct toyzero colormap

    This colormap uses three different colormaps for different magnitudes
    of normalized signal:
        - for signal outside of range RANGE_MID it uses `cmap_high`
        - for signal in range `RANGE_MID` it uses `cmap_mid`
        - for signal in range `RANGE_LOW` it uses `cmap_low`
    """
    def replace_colors(result, cmap, cspace, cspace_range, washout):
        mask = ((cspace > cspace_range[0]) & (cspace < cspace_range[1]))
        norm = (cspace_range[1] - cspace_range[0])

        result[mask, :] = cmap(0.5 + (cspace[mask] - 0.5) / norm) / washout

    cmap_low  = mpl.cm.get_cmap(cmap_low,  N_low)
    cmap_mid  = mpl.cm.get_cmap(cmap_mid,  N_mid)
    cmap_high = mpl.cm.get_cmap(cmap_high, N_high)

    colorspace = np.linspace(0, 1, N_high + N_low + N_mid)

    result = cmap_high(colorspace)
    replace_colors(result, cmap_mid, colorspace, RANGE_MID, washout_mid)
    replace_colors(result, cmap_low, colorspace, RANGE_LOW, washout_low)

    result[np.isclose(colorspace, 0.5), :] = 0

    return mpl.colors.ListedColormap(result)

def get_custom_cmap_norm(cmap, vrange, _log = False):
    """Main selector for custom colormaps"""

    if cmap == 'toyzero':
        return (get_toyzero_cmap(), ToyzeroNorm(vrange))

    return None

