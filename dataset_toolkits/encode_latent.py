import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import copy
import json
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from easydict import EasyDict as edict
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

from typing import Any, cast

import trellis.models as models
import trellis.modules.sparse as sp


torch.set_grad_enabled(False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--feat_model', type=str, default='dinov2_vitl14_reg',
                        help='Feature model')
    parser.add_argument('--enc_pretrained', type=str, default='JeffreyXiang/TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16',
                        help='Pretrained encoder model')
    parser.add_argument('--model_root', type=str, default='results',
                        help='Root directory of models')
    parser.add_argument('--enc_model', type=str, default=None,
                        help='Encoder model. if specified, use this model instead of pretrained model')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Checkpoint to load')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    opt = parser.parse_args()
    opt = edict(vars(opt))

    opt_any = cast(Any, opt)
    output_dir: str = str(opt_any.output_dir)

    if opt_any.enc_model is None:
        feat_model = str(opt_any.feat_model)
        enc_pretrained = str(opt_any.enc_pretrained)
        latent_name = f'{feat_model}_{enc_pretrained.split("/")[-1]}'
        encoder = models.from_pretrained(enc_pretrained).eval().cuda()
    else:
        feat_model = str(opt_any.feat_model)
        enc_model = str(opt_any.enc_model)
        ckpt = str(opt_any.ckpt)
        model_root = str(opt_any.model_root)
        latent_name = f'{feat_model}_{enc_model}_{ckpt}'
        cfg = edict(json.load(open(os.path.join(model_root, enc_model, 'config.json'), 'r')))
        cfg_any = cast(Any, cfg)
        encoder = getattr(models, cfg_any.models.encoder.name)(**cfg_any.models.encoder.args).cuda()
        ckpt_path = os.path.join(model_root, enc_model, 'ckpts', f'encoder_{ckpt}.pt')
        encoder.load_state_dict(torch.load(ckpt_path), strict=False)
        encoder.eval()
        print(f'Loaded model from {ckpt_path}')
    
    os.makedirs(os.path.join(output_dir, 'latents', latent_name), exist_ok=True)

    # get file list
    if os.path.exists(os.path.join(output_dir, 'metadata.csv')):
        metadata = pd.read_csv(os.path.join(output_dir, 'metadata.csv'))
    else:
        raise ValueError('metadata.csv not found')
    if opt_any.instances is not None:
        with open(str(opt_any.instances), 'r') as f:
            sha256s = [line.strip() for line in f]
        metadata = metadata[metadata['sha256'].isin(sha256s)]
    else:
        if opt_any.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= float(opt_any.filter_low_aesthetic_score)]
        feat_model = str(opt_any.feat_model)
        metadata = metadata[metadata[f'feature_{feat_model}'] == True]
        if f'latent_{latent_name}' in metadata.columns:
            metadata = metadata[metadata[f'latent_{latent_name}'] == False]

    start = len(metadata) * int(opt_any.rank) // int(opt_any.world_size)
    end = len(metadata) * (int(opt_any.rank) + 1) // int(opt_any.world_size)
    metadata = metadata[start:end]
    records = []
    
    # filter out objects that are already processed
    sha256s = list(metadata['sha256'].values)
    for sha256 in copy.copy(sha256s):
        if os.path.exists(os.path.join(output_dir, 'latents', latent_name, f'{sha256}.npz')):
            records.append({'sha256': sha256, f'latent_{latent_name}': True})
            sha256s.remove(sha256)

    # encode latents
    load_queue = Queue(maxsize=4)
    try:
        with ThreadPoolExecutor(max_workers=32) as loader_executor, \
            ThreadPoolExecutor(max_workers=32) as saver_executor:
            def loader(sha256):
                try:
                    feats = np.load(os.path.join(output_dir, 'features', str(opt_any.feat_model), f'{sha256}.npz'))
                    load_queue.put((sha256, feats))
                except Exception as e:
                    print(f"Error loading features for {sha256}: {e}")
            loader_executor.map(loader, sha256s)
            
            def saver(sha256, pack):
                save_path = os.path.join(output_dir, 'latents', latent_name, f'{sha256}.npz')
                np.savez_compressed(save_path, **pack)
                records.append({'sha256': sha256, f'latent_{latent_name}': True})
                
            for _ in tqdm(range(len(sha256s)), desc="Extracting latents"):
                sha256, feats = load_queue.get()
                feats = sp.SparseTensor(
                    feats = torch.from_numpy(feats['patchtokens']).float(),
                    coords = torch.cat([
                        torch.zeros(feats['patchtokens'].shape[0], 1).int(),
                        torch.from_numpy(feats['indices']).int(),
                    ], dim=1),
                ).cuda()
                latent = encoder(feats, sample_posterior=False)
                assert torch.isfinite(latent.feats).all(), "Non-finite latent"
                pack = {
                    'feats': latent.feats.cpu().numpy().astype(np.float32),
                    'coords': latent.coords[:, 1:].cpu().numpy().astype(np.uint8),
                }
                saver_executor.submit(saver, sha256, pack)
                
            saver_executor.shutdown(wait=True)
    except:
        print("Error happened during processing.")
        
    records_df = pd.DataFrame.from_records(records)
    records_df.to_csv(os.path.join(output_dir, f'latent_{latent_name}_{int(opt_any.rank)}.csv'), index=False)
