import json
import shutil
import os
import numpy as np
import torch

import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image

def update_json(filename, d):
    try:
        with open(filename, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {}
    for k in d:
        data[k] = d[k]
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

    


def collect_and_merge_results(root_dir):
    root_dir = os.path.abspath(root_dir)
    global_res_path = os.path.join(root_dir, "res.json")

    collected = {}

    # 1. Collect subdirectory results
    for name in os.listdir(root_dir):
        subdir = os.path.join(root_dir, name)
       
        if not os.path.isdir(subdir):
            continue

        res_path = os.path.join(subdir, "res.json")
        if os.path.isfile(res_path):
            with open(res_path, "r") as f:
                collected[name] = json.load(f)

    # 2. Load global res.json if exists
    if os.path.isfile(global_res_path):
        with open(global_res_path, "r") as f:
            global_res = json.load(f)
    else:
        global_res = {}

    # 3. Merge (non-destructive update)
    for method, method_dict in collected.items():
        if method not in global_res:
            global_res[method] = {}

        for k, v in method_dict.items():
            global_res[method][k] = v

    # 4. Write updated global res.json
    with open(global_res_path, "w") as f:
        json.dump(global_res, f, indent=2)

    # 5. Delete all subdirectories
    for name in os.listdir(root_dir):
        subdir = os.path.join(root_dir, name)
        
        if os.path.isdir(subdir):
            shutil.rmtree(subdir)





def print_stats(root_dir):
    root_dir = os.path.abspath(root_dir)
    global_res_path = os.path.join(root_dir, "res.json")
    with open(global_res_path, "r") as f:
        data = json.load(f)

    # Sort methods by descending FID
    metric = "fid"
    if metric == "clipscore":
        data = {k: data[k] for k in data if "clipscore" in data[k]}

    reverse = metric == "fid"

    sorted_methods = sorted(data.items(), key=lambda x: x[1][metric], reverse=reverse)

    # Print each method in order
    for method, metrics in sorted_methods:
        print(method, metrics[metric])



def save_overlay(
    image_pil,
    heatmap,
    out_path,
    alpha=0.45,
    cmap="hot",
    target_size=512,
):
    """
    image_pil: PIL.Image (512x512)
    heatmap: torch.Tensor or np.ndarray (64x64) or (1,64,64)
    """

    # --- to torch [1,1,H,W]
    if isinstance(heatmap, np.ndarray):
        heatmap = torch.from_numpy(heatmap)
    if heatmap.dim() == 2:
        heatmap = heatmap.unsqueeze(0).unsqueeze(0)
    elif heatmap.dim() == 3:
        heatmap = heatmap.unsqueeze(0)

    heatmap = heatmap.float()

    # --- resize to image size
    heatmap = F.interpolate(
        heatmap,
        size=(target_size, target_size),
        mode="bilinear",
        align_corners=False,
    )[0, 0]

    # --- normalize for visualization
    heatmap = heatmap - heatmap.min()
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    heatmap = heatmap.cpu().numpy()

    # --- plot overlay
    plt.figure(figsize=(5, 5))
    plt.imshow(image_pil)
    plt.imshow(heatmap, cmap=cmap, alpha=alpha)
    plt.axis("off")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()