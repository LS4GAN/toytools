"""Various helper functions to construct toyzero datasets."""

from .simple_toyzero          import SimpleToyzeroDataset
from .presimple_toyzero       import PreSimpleToyzeroDataset
from .presimple_toyzero_v1    import PreSimpleToyzeroDatasetV1
from .preunaligned_toyzero    import PreUnalignedToyzeroDataset
from .precropped_toyzero      import PreCroppedToyzeroDataset
from .precropped_toyzero_v1   import PreCroppedToyzeroDatasetV1

DATASET_DICT = {
    'toyzero-simple'        : SimpleToyzeroDataset,
    'toyzero-presimple'     : PreSimpleToyzeroDataset,
    'toyzero-preunaligned'  : PreUnalignedToyzeroDataset,
    'toyzero-precropped'    : PreCroppedToyzeroDataset,
    'toyzero-precropped-v1' : PreCroppedToyzeroDatasetV1,
    'toyzero-presimple-v1'  : PreSimpleToyzeroDatasetV1,
}

def get_toyzero_dataset(name, path, **data_args):
    """Return toyzero dataset based on its name"""

    if name not in DATASET_DICT:
        raise ValueError("Unknown dataset name: %s" % name)

    return DATASET_DICT[name](path, **data_args)

