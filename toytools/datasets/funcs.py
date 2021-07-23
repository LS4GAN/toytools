"""Various helper functions to construct toyzero datasets."""

from .simple_toyzero    import SimpleToyzeroDataset
from .presimple_toyzero import PreSimpleToyzeroDataset

def get_toyzero_dataset(name, path, **data_args):
    """Return toyzero dataset based on its name"""

    if name == 'toyzero-simple':
        return SimpleToyzeroDataset(path, **data_args)

    if name == 'toyzero-presimple':
        return PreSimpleToyzeroDataset(path, **data_args)

    raise ValueError("Unknown toyzero dataset name: %s" % name)

