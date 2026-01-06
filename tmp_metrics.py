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
import random

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



if __name__ == "__main__":
    gen_folders = ["uncertaintity_maps/SDXL/basic/coco/", "uncertaintity_maps/1.5v/basic/coco/"]
    
    real_dir = "datasets/coco/val2014"
    indices = range(30000)
    n_remove = int(len(indices) * 0.16)         # how many to remove
    to_remove = set(random.sample(indices, n_remove))  # pick exactly 16% randomly
    filtered = [x for x in indices if x not in to_remove]

   

    for fake_dir in gen_folders:
        fid_score = compute_fid_custom(fake_dir, real_dir,file_indices = filtered)
        print(f"FID: {fid_score:.4f}")


