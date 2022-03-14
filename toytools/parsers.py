"""Various common parsers """

import json
import re
from typing import Optional

from .datasets.funcs import DATASET_DICT

def add_dataset_parser(parser):
    # pylint: disable=missing-function-docstring
    datasets = list(DATASET_DICT.keys())

    parser.add_argument(
        '--dataset',
        choices = datasets,
        default = datasets[0],
        dest    = 'dataset',
        help    = 'type of the toyzero dataset to use',
        type    = str,
    )

def add_dataset_args_parser(parser):
    # pylint: disable=missing-function-docstring
    parser.add_argument(
        '--data_args',
        default = {},
        dest    = 'data_args',
        help    = (
            "JSON dict of data_args parameters. "
            "Example: '{ \"seed\": 1, \"val_size\": 0.4  }'"
        ),
        type    = json.loads,
    )

def add_colormap_parser(parser):
    # pylint: disable=missing-function-docstring
    parser.add_argument(
        '--cmap',
        default = 'seismic',
        dest    = 'cmap',
        help    = 'plot colormap',
        type    = str,
    )

def add_log_norm_parser(parser):
    # pylint: disable=missing-function-docstring
    parser.add_argument(
        '--log',
        action  = 'store_true',
        dest    = 'log',
        help    = 'use logarithmic colormap normalization',
    )

def add_symmetric_norm_parser(parser):
    # pylint: disable=missing-function-docstring
    parser.add_argument(
        '--symmetric',
        action  = 'store_true',
        dest    = 'symmetric',
        help    = 'assume that image values are symmetric around zero',
    )

def add_index_range_parser(parser):
    # pylint: disable=missing-function-docstring
    parser.add_argument(
        '-i',
        default = None,
        dest    = 'index',
        help    = \
            'index or range of indices [first:last) of the samples to select',
        type    = str,
    )

def parse_index_range(index : Optional[str], full_len : int) -> range:
    # pylint: disable=missing-function-docstring
    if index is None:
        return range(full_len)

    if index.isnumeric():
        i = int(index)
        return range(i, i+1)

    range_re = re.compile(r'(\d*):(\d*)')
    match    = range_re.match(index)

    if not match:
        raise ValueError(f"Failed to parse index range: '{index}'")

    first, last = [ (int(x) if x != '' else None) for x in match.groups() ]
    index_slice = slice(first, last)

    return range(*index_slice.indices(full_len))

