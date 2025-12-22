import argparse
from cleanfid.cleanfid import fid as cfid
from pathlib import Path
import os
from datetime import datetime
from PIL import Image, ImageFile
from concurrent.futures import ProcessPoolExecutor as Pool
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch

from torchvision.transforms import functional as F
from torchmetrics.image.fid import FrechetInceptionDistance




from cleanfid.cleanfid import fid

def compute_fid(dir_fake, dir_real, batch_size=256, num_workers=4):
    score = fid.compute_fid(
        dir_fake,
        dir_real,
        batch_size=batch_size,
        num_workers=num_workers,
        mode="clean",
        model_name="inception_v3",
    )
    return score


def compute_fid_custom(dir_fake, dir_real, batch_size=256, num_workers=4, file_indices = None):
    

    score = fid.compute_fid(
        dir_fake,
        dir_real,
        batch_size=batch_size,
        num_workers=num_workers,
        mode="clean",
        model_name="inception_v3",
        file_indices = file_indices
        
    )
    return score


'''
if __name__ == "__main__":
    fake_dir = "uncertaintity_maps/1.5v/basic/coco"
    real_dir = "datasets/coco/val2014"

    fid_score = compute_fid(fake_dir, real_dir)
    print(f"FID: {fid_score:.4f}")'''

'''

import os
import torch
import dnnlib
import hashlib

def compute_feature_stats_for_dataset(opts, detector_url, detector_kwargs, rel_lo=0, rel_hi=1, batch_size=64, data_loader_kwargs=None, max_items=None, **stats_kwargs):
    ##added For COCO_val256 FID computation
    #opts.dataset_kwargs.resolution=256
    ###
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
    if data_loader_kwargs is None:
        data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)

    # Try to lookup from cache.
    cache_file = None
    if opts.cache:
        # Choose cache file name.
        args = dict(dataset_kwargs=opts.dataset_kwargs, detector_url=detector_url, detector_kwargs=detector_kwargs, stats_kwargs=stats_kwargs)
        md5 = hashlib.md5(repr(sorted(args.items())).encode('utf-8'))
        cache_tag = f'{dataset.name}-{get_feature_detector_name(detector_url)}-{md5.hexdigest()}'
        cache_file = dnnlib.make_cache_dir_path('gan-metrics', cache_tag + '.pkl')

        # Check if the file exists (all processes must agree).
        flag = os.path.isfile(cache_file) if opts.rank == 0 else False
        if opts.num_gpus > 1:
            flag = torch.as_tensor(flag, dtype=torch.float32, device=opts.device)
            torch.distributed.broadcast(tensor=flag, src=0)
            flag = (float(flag.cpu()) != 0)

        # Load.
        if flag:
            return FeatureStats.load(cache_file)

    # Initialize.
    num_items = len(dataset)
    if max_items is not None:
        num_items = min(num_items, max_items)
    stats = FeatureStats(max_items=num_items, **stats_kwargs)
    progress = opts.progress.sub(tag='dataset features', num_items=num_items, rel_lo=rel_lo, rel_hi=rel_hi)
    detector = get_feature_detector(url=detector_url, device=opts.device, num_gpus=opts.num_gpus, rank=opts.rank, verbose=progress.verbose)

    # Main loop.
    item_subset = [(i * opts.num_gpus + opts.rank) % num_items for i in range((num_items - 1) // opts.num_gpus + 1)]
    for images, _labels in torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset, batch_size=batch_size, **data_loader_kwargs):
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        features = detector(images.to(opts.device), **detector_kwargs)
        stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)
        progress.update(stats.num_items)

    # Save to cache.
    if cache_file is not None and opts.local_rank == 0:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        temp_file = cache_file + '.' + uuid.uuid4().hex
        stats.save(temp_file)
        os.replace(temp_file, cache_file) # atomic
    return stats

def compute_distances(row_features, col_features, num_gpus, rank, col_batch_size):
    assert 0 <= rank < num_gpus
    num_cols = col_features.shape[0]
    num_batches = ((num_cols - 1) // col_batch_size // num_gpus + 1) * num_gpus
    col_batches = torch.nn.functional.pad(col_features, [0, 0, 0, -num_cols % num_batches]).chunk(num_batches)
    dist_batches = []
    for col_batch in col_batches[rank :: num_gpus]:
        dist_batch = torch.cdist(row_features.unsqueeze(0), col_batch.unsqueeze(0))[0]
        for src in range(num_gpus):
            dist_broadcast = dist_batch.clone()
            if num_gpus > 1:
                torch.distributed.broadcast(dist_broadcast, src=src)
            dist_batches.append(dist_broadcast.cpu() if rank == 0 else None)
    return torch.cat(dist_batches, dim=1)[:, :num_cols] if rank == 0 else None

#----------------------------------------------------------------------------

def compute_pr(opts, max_real, num_gen, nhood_size, row_batch_size, col_batch_size):
    
    #detector_url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    detector_url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt'
    
    #IsGoogle3
    #detector_url = opts.metric_pt_path
    
    detector_kwargs = dict(return_features=True)

    real_features = compute_feature_stats_for_dataset(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=0, capture_all=True, max_items=max_real).get_all_torch().to(torch.float16).to(opts.device)

    gen_features =  compute_feature_stats_for_generator(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=1, capture_all=True, max_items=num_gen).get_all_torch().to(torch.float16).to(opts.device)

    results = dict()
    for name, manifold, probes in [('precision', real_features, gen_features), ('recall', gen_features, real_features)]:
        kth = []
        for manifold_batch in manifold.split(row_batch_size):
            dist = compute_distances(row_features=manifold_batch, col_features=manifold, num_gpus=opts.num_gpus, rank=opts.rank, col_batch_size=col_batch_size)
            kth.append(dist.to(torch.float32).kthvalue(nhood_size + 1).values.to(torch.float16) if opts.rank == 0 else None)
        kth = torch.cat(kth) if opts.rank == 0 else None
        pred = []
        for probes_batch in probes.split(row_batch_size):
            dist = compute_distances(row_features=probes_batch, col_features=manifold, num_gpus=opts.num_gpus, rank=opts.rank, col_batch_size=col_batch_size)
            pred.append((dist <= kth).any(dim=1) if opts.rank == 0 else None)
        results[name] = float(torch.cat(pred).to(torch.float32).mean() if opts.rank == 0 else 'nan')
    return results['precision'], results['recall']
    '''