"""A collection of pytorch like Datasets to handle toyzero data"""

import importlib
import importlib.util

from .funcs                import get_toyzero_dataset
from .simple_toyzero       import SimpleToyzeroDataset
from .presimple_toyzero    import PreSimpleToyzeroDataset
from .preunaligned_toyzero import PreUnalignedToyzeroDataset

__all__ = [
    'SimpleToyzeroDataset', 'PreSimpleToyzeroDataset',
    'PreUnalignedToyzeroDataset', 'get_toyzero_dataset',
]

# NOTE: it imports torch-dependent functions conditionally on pytorch presence
if importlib.util.find_spec("torch") is not None:
    if importlib.util.find_spec("torch.utils") is not None:
        from .torch_funcs import get_toyzero_dataset_torch
        __all__.append('get_toyzero_dataset_torch')

