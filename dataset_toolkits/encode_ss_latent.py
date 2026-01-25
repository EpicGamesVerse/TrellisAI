import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import copy
import json
import argparse
import torch
import numpy as np
import pandas as pd
import utils3d
from tqdm import tqdm
from easydict import EasyDict as edict
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import trellis.models as models

from typing import Any, cast


torch.set_grad_enabled(False)


def get_voxels(instance: str, output_dir: str, resolution: int) -> torch.Tensor:
    position = utils3d.io.read_ply(os.path.join(output_dir, 'voxels', f'{instance}.ply'))[0]
    coords = ((torch.tensor(position) + 0.5) * resolution).int().contiguous()
    ss = torch.zeros(1, resolution, resolution, resolution, dtype=torch.long)
    ss[:, coords[:, 0], coords[:, 1], coords[:, 2]] = 1
    return ss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--enc_pretrained', type=str, default='JeffreyXiang/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16',
                        help='Pretrained encoder model')
    parser.add_argument('--model_root', type=str, default='results',
                        help='Root directory of models')
    parser.add_argument('--enc_model', type=str, default=None,
                        help='Encoder model. if specified, use this model instead of pretrained model')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Checkpoint to load')
    parser.add_argument('--resolution', type=int, default=64,
                        help='Resolution')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    opt = parser.parse_args()
    opt = edict(vars(opt))

    opt_any = cast(Any, opt)
    output_dir: str = str(opt_any.output_dir)
    resolution: int = int(opt_any.resolution)

    if opt_any.enc_model is None:
        enc_pretrained = str(opt_any.enc_pretrained)
        latent_name = f'{enc_pretrained.split("/")[-1]}'
        encoder = models.from_pretrained(enc_pretrained).eval().cuda()
    else:
        enc_model = str(opt_any.enc_model)
        ckpt = str(opt_any.ckpt)
        model_root = str(opt_any.model_root)
        latent_name = f'{enc_model}_{ckpt}'
        cfg = edict(json.load(open(os.path.join(model_root, enc_model, 'config.json'), 'r')))
        cfg_any = cast(Any, cfg)
        encoder = getattr(models, cfg_any.models.encoder.name)(**cfg_any.models.encoder.args).cuda()
        ckpt_path = os.path.join(model_root, enc_model, 'ckpts', f'encoder_{ckpt}.pt')
        encoder.load_state_dict(torch.load(ckpt_path), strict=False)
        encoder.eval()
        print(f'Loaded model from {ckpt_path}')
    
    os.makedirs(os.path.join(output_dir, 'ss_latents', latent_name), exist_ok=True)

    # get file list
    if os.path.exists(os.path.join(output_dir, 'metadata.csv')):
        metadata = pd.read_csv(os.path.join(output_dir, 'metadata.csv'))
    else:
        raise ValueError('metadata.csv not found')
    if opt_any.instances is not None:
        with open(str(opt_any.instances), 'r') as f:
            instances = f.read().splitlines()
        metadata = metadata[metadata['sha256'].isin(instances)]
    else:
        if opt_any.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= float(opt_any.filter_low_aesthetic_score)]
        metadata = metadata[metadata['voxelized'] == True]
        if f'ss_latent_{latent_name}' in metadata.columns:
            metadata = metadata[metadata[f'ss_latent_{latent_name}'] == False]

    start = len(metadata) * int(opt_any.rank) // int(opt_any.world_size)
    end = len(metadata) * (int(opt_any.rank) + 1) // int(opt_any.world_size)
    metadata = metadata[start:end]
    records = []
    
    # filter out objects that are already processed
    sha256s = list(metadata['sha256'].values)
    for sha256 in copy.copy(sha256s):
        if os.path.exists(os.path.join(output_dir, 'ss_latents', latent_name, f'{sha256}.npz')):
            records.append({'sha256': sha256, f'ss_latent_{latent_name}': True})
            sha256s.remove(sha256)

    # encode latents
    load_queue = Queue(maxsize=4)
    try:
        with ThreadPoolExecutor(max_workers=32) as loader_executor, \
            ThreadPoolExecutor(max_workers=32) as saver_executor:
            def loader(sha256):
                try:
                    ss = get_voxels(sha256, output_dir=output_dir, resolution=resolution)[None].float()
                    load_queue.put((sha256, ss))
                except Exception as e:
                    print(f"Error loading features for {sha256}: {e}")
            loader_executor.map(loader, sha256s)
            
            def saver(sha256, pack):
                save_path = os.path.join(output_dir, 'ss_latents', latent_name, f'{sha256}.npz')
                np.savez_compressed(save_path, **pack)
                records.append({'sha256': sha256, f'ss_latent_{latent_name}': True})
                
            for _ in tqdm(range(len(sha256s)), desc="Extracting latents"):
                sha256, ss = load_queue.get()
                ss = ss.cuda().float()
                latent = encoder(ss, sample_posterior=False)
                assert torch.isfinite(latent).all(), "Non-finite latent"
                pack = {
                    'mean': latent[0].cpu().numpy(),
                }
                saver_executor.submit(saver, sha256, pack)
                
            saver_executor.shutdown(wait=True)
    except:
        print("Error happened during processing.")
        
    records_df = pd.DataFrame.from_records(records)
    records_df.to_csv(os.path.join(output_dir, f'ss_latent_{latent_name}_{int(opt_any.rank)}.csv'), index=False)
