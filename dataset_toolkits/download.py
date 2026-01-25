import os
import copy
import sys
import importlib
import argparse
import pandas as pd
from easydict import EasyDict as edict

from typing import Any, cast

if __name__ == '__main__':
    dataset_utils = importlib.import_module(f'datasets.{sys.argv[1]}')

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    dataset_utils.add_args(parser)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))

    opt_any = cast(Any, opt)
    output_dir: str = str(opt_any.output_dir)
    rank: int = int(opt_any.rank)
    world_size: int = int(opt_any.world_size)
    opt_dict: dict[str, Any] = dict(opt_any)

    os.makedirs(output_dir, exist_ok=True)

    # get file list
    if not os.path.exists(os.path.join(output_dir, 'metadata.csv')):
        raise ValueError('metadata.csv not found')
    metadata = pd.read_csv(os.path.join(output_dir, 'metadata.csv'))
    if opt_any.instances is None:
        if opt_any.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= float(opt_any.filter_low_aesthetic_score)]
        if 'local_path' in metadata.columns:
            metadata = metadata[metadata['local_path'].isna()]
    else:
        instances_arg = str(opt_any.instances)
        if os.path.exists(instances_arg):
            with open(instances_arg, 'r') as f:
                instances = f.read().splitlines()
        else:
            instances = instances_arg.split(',')
        metadata = metadata[metadata['sha256'].isin(instances)]

    start = len(metadata) * rank // world_size
    end = len(metadata) * (rank + 1) // world_size
    metadata = metadata[start:end]
                
    print(f'Processing {len(metadata)} objects...')

    # process objects
    downloaded = dataset_utils.download(metadata, **opt_dict)
    downloaded.to_csv(os.path.join(output_dir, f'downloaded_{rank}.csv'), index=False)
