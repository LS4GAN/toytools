"""Various helper functions to construct toyzero datasets."""

from .simple_toyzero          import SimpleToyzeroDataset
from .presimple_toyzero       import PreSimpleToyzeroDataset
from .presimple_toyzero_v1    import PreSimpleToyzeroDatasetV1
from .preunaligned_toyzero    import PreUnalignedToyzeroDataset
from .precropped_toyzero      import PreCroppedToyzeroDataset
from .precropped_toyzero_v1   import PreCroppedToyzeroDatasetV1

def get_toyzero_dataset(name, path, **data_args):
    """Return toyzero dataset based on its name"""

    if name == 'toyzero-simple':
        return SimpleToyzeroDataset(path, **data_args)

    if name == 'toyzero-presimple':
        return PreSimpleToyzeroDataset(path, **data_args)

    if name == 'toyzero-preunaligned':
        return PreUnalignedToyzeroDataset(path, **data_args)

    if name == 'toyzero-precropped':
        return PreCroppedToyzeroDataset(path, **data_args)

    if name == 'toyzero-precropped-v1':
        return PreCroppedToyzeroDatasetV1(path, **data_args)

    if name == 'toyzero-presimple-v1':
        return PreSimpleToyzeroDatasetV1(path, **data_args)

    raise ValueError("Unknown toyzero dataset name: %s" % name)

