import json
import shutil
import os
import numpy as np
import torch
import scipy.stats as stats
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from scipy.ndimage import distance_transform_edt

from PIL import ImageFont

FONT_SIZE = 32  # try 28–40 depending on resolution

FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",  # linux
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

for p in FONT_PATHS:
    try:
        FONT = ImageFont.truetype(p, FONT_SIZE)
        break
    except:
        FONT = None

if FONT is None:
    FONT = ImageFont.load_default()


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

    
def update_json2(filename, d, debug=False):
    import json

    try:
        with open(filename, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {}


    if debug:
        print(data)
        print(d.items())
        exit(1)
    for k, v in d.items():
        if k not in data:
            data[k] = {}

        for subkey, subval in v.items():
            # If this is a dict (score / score_aux), merge by key
            if isinstance(subval, dict):
                if subkey not in data[k]:
                    data[k][subkey] = {}
                data[k][subkey].update(subval)
            else:
                data[k][subkey] = subval

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
    metric = "pick_score" # "aes_score"  #     # "cmmd" "hpsv2_score" 
    #if metric == "clipscore":
    data = {k: data[k] for k in data if metric in data[k]}

    reverse = metric == "fid"

    sorted_methods = sorted(data.items(), key=lambda x: x[1][metric], reverse=reverse)

    # Print each method in order
    for method, metrics in sorted_methods:
        if "above" in method or ("count" in method and "_pr" not in method):
            continue
        #if "ASCED" not in method:
        #    continue
        print(method, metrics[metric])


import matplotlib.cm as cm

def overlay_to_pil_simple(
    image_pil,
    heatmap,
    alpha=0.5,
    cmap="hot",
    target_size=512,
):
    # --- heatmap to torch [H,W]
    if isinstance(heatmap, np.ndarray):
        heatmap = torch.from_numpy(heatmap)
    if heatmap.dim() == 3:
        heatmap = heatmap.squeeze(0)
    if heatmap.dim() == 4:
        heatmap = heatmap[0, 0]

    heatmap = heatmap.float()

    # --- resize heatmap
    heatmap = heatmap.unsqueeze(0).unsqueeze(0)
    heatmap = F.interpolate(
        heatmap,
        size=(target_size, target_size),
        mode="bilinear",
        align_corners=False,
    )[0, 0]

    # --- normalize
    heatmap = heatmap - heatmap.min()
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    heatmap = heatmap.clamp(0, 1).cpu().numpy()

    # --- apply colormap (RGBA)
    cmap_fn = cm.get_cmap(cmap)
    heatmap_rgba = cmap_fn(heatmap)  # H,W,4
    heatmap_img = Image.fromarray(
        (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
    )

    # --- resize image
    image_pil = image_pil.resize((target_size, target_size))

    # --- alpha blend
    return Image.blend(image_pil, heatmap_img, alpha)



import csv
from PIL import ImageDraw, ImageFont
import random

def compose_row(reference, overlays_dict, out_path, sample_id, csv_writer):
    """
    overlays_dict: {"sup": PIL, "cumperc": PIL, "asced": PIL}
    """
    methods = list(overlays_dict.keys())
    random.shuffle(methods)

    ordered_imgs = [overlays_dict[m] for m in methods]

    W, H = reference.size
    gap_big = 40     # ref → visualizations
    gap_small = 10   # between visualizations
    label_space = 40

    total_w = W * 3 + gap_big + 1 * gap_small
    total_h = H + label_space

    canvas = Image.new("RGB", (total_w, total_h), "white")
    draw = ImageDraw.Draw(canvas)

    x = 0
    canvas.paste(reference, (x, 0))
    draw.text((x + W // 2 - 70, H + 2), "Reference", fill="black",  font=FONT)
    x += W + gap_big

    labels = ["(A)", "(B)", "(C)"]
    for lbl, img in zip(labels, ordered_imgs):
        canvas.paste(img, (x, 0))
        draw.text((x + W // 2 - 10, H + 2), lbl, fill="black", font=FONT)
        x += W + gap_small

    canvas.save(out_path)

    csv_writer.writerow({
        "sample_id": sample_id,
        "A": methods[0],
        "B": methods[1],
        #"C": methods[2],
    })


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

    #threshold = torch.quantile(heatmap, 90 / 100.0)
    #heatmap = torch.where(heatmap >= threshold, heatmap, torch.zeros_like(heatmap))

    heatmap = heatmap.cpu().numpy()

    # --- plot overlay
    plt.figure(figsize=(5, 5))
    plt.imshow(image_pil)
    plt.imshow(heatmap, cmap=cmap, alpha=alpha)
    plt.axis("off")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()



def prepare_heatmap(heatmap, target_size):
    if isinstance(heatmap, np.ndarray):
        heatmap = torch.from_numpy(heatmap)

    if heatmap.dim() == 2:
        heatmap = heatmap.unsqueeze(0).unsqueeze(0)
    elif heatmap.dim() == 3:
        heatmap = heatmap.unsqueeze(0)

    heatmap = heatmap.float()

    heatmap = F.interpolate(
        heatmap,
        size=(target_size, target_size),
        mode="bilinear",
        align_corners=False,
    )[0, 0]

   

    return heatmap.cpu().numpy()

def save_overlay_row(
    image_pil,
    heatmaps,
    out_path,
    alpha=0.75,
    cmap="hot",
    target_size=512,
    gap=0.05,
):
    """
    heatmaps: list of torch.Tensor / np.ndarray
    """

    fig, axes = plt.subplots(1, len(heatmaps), figsize=(5 * len(heatmaps), 5))

    if len(heatmaps) == 1:
        axes = [axes]

    for ax, heatmap in zip(axes, heatmaps):
        hm = prepare_heatmap(heatmap, target_size)

        ax.imshow(image_pil)
        ax.imshow(hm, cmap=cmap, alpha=alpha)
        ax.axis("off")

    plt.subplots_adjust(wspace=gap)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()

def playground(args):

    subdirs = sorted([d for d in os.listdir(args.output_dir) if os.path.isdir(os.path.join(args.output_dir, d))], 
                        key=lambda x: int(x))
    for subdir in subdirs:
        subdir_path = os.path.join(args.output_dir, subdir)
        with open(f"{subdir_path}/prompt.txt", "r") as f:
            line = f.readline().strip()
        if "bathroom" in line:
            print(f"{subdir}:{line}")
            print("\n")

    exit(1)
    motivation_dir = "tmp/motivation"
    good_dir = f"{motivation_dir}/good"
    bad_dir = f"{motivation_dir}/bad"

    os.makedirs(motivation_dir, exist_ok=True)
    os.makedirs(good_dir, exist_ok=True)
    os.makedirs(bad_dir, exist_ok=True)

    method = "basic_pr_globalTimestep_12$-5"

    dir_subset = "visualizations/compare/coco/1.5v/best_worst/subset"

    subdirs = sorted([d for d in os.listdir(dir_subset) if os.path.isdir(os.path.join(dir_subset, d))], 
                        key=lambda x: int(x))

   

    with open(f"visualizations/compare/coco/1.5v/best_worst/res.json") as f:
        data = json.load(f)

    data = data[method]
    sorted_samples = sorted(data.items(), key=lambda x: x[1], reverse=True)

    worst = sorted_samples[:100]
    best = sorted_samples[100:]
    batch_size = 8
    complete_list = [worst, best]

    for i_s, samples in enumerate(complete_list):
        sample_idx = 0
        curr_dir = bad_dir if i_s ==0 else good_dir
        for batch_start in range(0, len(samples), batch_size):
            batch_end = min(batch_start + batch_size, len(samples))
            batch_items = samples[batch_start:batch_end]
            prompts = []
            sample_id_lst = []
            for item in batch_items:
                sample_id      = item[0]
                prompt_file    = f"{dir_subset}/{sample_id}/prompt.txt"
                
                with open(prompt_file, "r") as f:
                    line = f.readline().strip() 
                prompts.append(line)
                sample_id_lst.append(sample_id)
            
            

            generator = torch.Generator(device="cuda").manual_seed(2024)
            output = args.pipe(prompts, apply_uc = True, apply_uc_on_all_timesteps=True, return_aux = True, generator= generator, return_mid_reps = True,  apply_var_method_uc=False)
            images = output[0].images
            uncertainty_maps = output[1]["uncertainty_maps"]
            aux_list         = output[1]["aux_list"]

            for idx in range(len(images)):
                images[idx].save(f"{curr_dir}/output{sample_id_lst[idx]}.jpg", quality=95)
                
                update_json2(f"{curr_dir}/res.json", {sample_id_lst[idx] : {"prompt": prompts[idx]} })



            timesteps = sorted(uncertainty_maps.keys(), reverse=True)
   
            example_ts = timesteps[0]
            last_layer = sorted(uncertainty_maps[example_ts].keys())[-1]

            num_samples = int(uncertainty_maps[example_ts][last_layer].shape[0] / 2)
            
            for heatmap_idx in range(num_samples):
                for map_idx in range(1):
                    for row_idx, ts in enumerate(timesteps):
                        uncertainty = uncertainty_maps[ts][last_layer].chunk(2)[map_idx][heatmap_idx].squeeze(0)
                        latent = aux_list[row_idx][heatmap_idx]

                        update_json2(f"{curr_dir}/res.json", {sample_id_lst[heatmap_idx] : {"score": {ts: float(uncertainty.sum().item())}} })

                        update_json2(f"{curr_dir}/res.json", {sample_id_lst[heatmap_idx] : {"score_aux": {ts: float(latent.sum().item())}} })
            





import math
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
def load_mean_and_ci(json_path, key_name):
    with open(json_path, "r") as f:
        data = json.load(f)

    
    agg = defaultdict(list)
    for id_dict in data.values():
        for t, v in id_dict[key_name].items():
            agg[int(t)].append(v)

    timesteps = list(agg.keys())[::5]

    mean = []
    low = []
    high = []

    
    for t in timesteps:
        vals = agg[t]
        n = len(vals)
        mu = sum(vals) / n
        var = sum((x - mu) ** 2 for x in vals) / (n - 1)
        se = math.sqrt(var) / math.sqrt(n)

        mean.append(mu)
        low.append(mu - 1.96 * se)
        high.append(mu + 1.96 * se)
    

    return timesteps, mean, low, high




def compute_discrimination_metrics(json_path_bad, json_path_good, key_name):
    """
    Compute scale-invariant metrics for discrimination between bad and good.
    
    Returns:
        dict with metrics for each timestep
    """
    # Load raw data
    with open(json_path_bad, "r") as f:
        data_bad = json.load(f)
    with open(json_path_good, "r") as f:
        data_good = json.load(f)
    
    # Aggregate by timestep
    agg_bad = defaultdict(list)
    agg_good = defaultdict(list)
    
    for id_dict in data_bad.values():
        for t, v in id_dict[key_name].items():
            agg_bad[int(t)].append(v)
    
    for id_dict in data_good.values():
        for t, v in id_dict[key_name].items():
            agg_good[int(t)].append(v)
    
    timesteps = sorted(set(agg_bad.keys()) & set(agg_good.keys()))
    
    metrics = {
        'timesteps': timesteps,
        'cohens_d': [],
        'effect_size': [],
        'separation': [],
        'auc_proxy': []
    }
    
    for t in timesteps:
        bad_vals = np.array(agg_bad[t])
        good_vals = np.array(agg_good[t])
        
        # Cohen's d (standardized mean difference)
        mean_bad = np.mean(bad_vals)
        mean_good = np.mean(good_vals)
        pooled_std = np.sqrt((np.var(bad_vals, ddof=1) + np.var(good_vals, ddof=1)) / 2)
        cohens_d = (mean_bad - mean_good) / pooled_std
        
        # Effect size (difference in means relative to pooled range)
        pooled_range = max(np.max(bad_vals), np.max(good_vals)) - min(np.min(bad_vals), np.min(good_vals))
        effect_size = (mean_bad - mean_good) / pooled_range if pooled_range > 0 else 0
        
        # Separation metric (non-overlapping ratio)
        min_bad, max_bad = np.min(bad_vals), np.max(bad_vals)
        min_good, max_good = np.min(good_vals), np.max(good_vals)
        overlap = max(0, min(max_bad, max_good) - max(min_bad, min_good))
        total_range = max(max_bad, max_good) - min(min_bad, min_good)
        separation = 1 - (overlap / total_range) if total_range > 0 else 0
        
        # AUC proxy (probability bad > good)
        # This estimates P(X_bad > X_good) for random samples
        comparisons = bad_vals[:, None] > good_vals[None, :]
        auc_proxy = np.mean(comparisons)
        
        metrics['cohens_d'].append(cohens_d)
        metrics['effect_size'].append(effect_size)
        metrics['separation'].append(separation)
        metrics['auc_proxy'].append(auc_proxy)
    
    return metrics


def plot_comparison_metrics(ROOT, method_names=["score", "score_aux"]):
    """
    Plot discrimination metrics for different methods.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Method Discrimination Comparison (Scale-Invariant Metrics)", fontsize=14)
    
    colors = ["blue", "orange", "red", "green"]
    
    all_metrics = {}
    for i, method in enumerate(method_names):
        metrics = compute_discrimination_metrics(
            ROOT / "bad" / "res.json",
            ROOT / "good" / "res.json",
            method
        )
        all_metrics[method] = metrics
        
        # Plot Cohen's d
        axes[0, 0].plot(metrics['timesteps'], metrics['cohens_d'], 
                       color=colors[i], label=method, marker='o', markersize=3)
        
        # Plot AUC proxy
        axes[0, 1].plot(metrics['timesteps'], metrics['auc_proxy'], 
                       color=colors[i], label=method, marker='o', markersize=3)
        
        # Plot separation
        axes[1, 0].plot(metrics['timesteps'], metrics['separation'], 
                       color=colors[i], label=method, marker='o', markersize=3)
        
        # Plot effect size
        axes[1, 1].plot(metrics['timesteps'], metrics['effect_size'], 
                       color=colors[i], label=method, marker='o', markersize=3)
    
    # Configure subplots
    axes[0, 0].set_title("Cohen's d (Standardized Mean Difference)")
    axes[0, 0].set_xlabel("Timestep")
    axes[0, 0].set_ylabel("Cohen's d")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    
    axes[0, 1].set_title("AUC Proxy (P(bad > good))")
    axes[0, 1].set_xlabel("Timestep")
    axes[0, 1].set_ylabel("Probability")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0.5, color='black', linestyle='--', linewidth=0.5)
    axes[0, 1].set_ylim([0, 1])
    
    axes[1, 0].set_title("Separation (1 - Overlap Ratio)")
    axes[1, 0].set_xlabel("Timestep")
    axes[1, 0].set_ylabel("Separation")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1])
    
    axes[1, 1].set_title("Effect Size (Normalized Difference)")
    axes[1, 1].set_xlabel("Timestep")
    axes[1, 1].set_ylabel("Effect Size")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("method_comparison_metricsNEW.png", dpi=300)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY: Average Metrics Across All Timesteps")
    print("="*60)
    for method in method_names:
        m = all_metrics[method]
        print(f"\n{method}:")
        print(f"  Cohen's d (avg):        {np.mean(m['cohens_d']):.3f}")
        print(f"  AUC Proxy (avg):        {np.mean(m['auc_proxy']):.3f}")
        print(f"  Separation (avg):       {np.mean(m['separation']):.3f}")
        print(f"  Effect Size (avg):      {np.mean(m['effect_size']):.3f}")
    
    return all_metrics



def reorganize_samples():
    INPUT_JSON = "tmp_motivation/res.json"
    bad_dir = f"tmp_motivation/bad"
    good_dir = f"tmp_motivation/good"
    os.makedirs(good_dir, exist_ok = True)
    os.makedirs(bad_dir, exist_ok = True)

    # Load original data
    with open(INPUT_JSON, "r") as f:
        data = json.load(f)

    # Compute scores
    id_scores = []
    for id_, content in data.items():
        score_dict = content["score"]  # dict: {number: float}
        total_score = sum(score_dict.values())
        id_scores.append((id_, total_score))

    # Sort by score (ascending)
    id_scores.sort(key=lambda x: x[1])

    # Split evenly
    mid = len(id_scores) // 2
    low_ids = id_scores[:mid]
    high_ids = id_scores[mid:]

    

    # Build output dicts (preserving full content)
    low_data = {id_: data[id_] for id_, _ in low_ids}
    high_data = {id_: data[id_] for id_, _ in reversed(high_ids)}

    # Save results
    with open(f"{good_dir}/res.json", "w") as f:
        json.dump(low_data, f, indent=2)

    with open(f"{bad_dir}/res.json", "w") as f:
        json.dump(high_data, f, indent=2)
    
    for rank, (id_, _) in enumerate(low_ids, start=1):
        subdir = os.path.join(good_dir, f"{rank:04d}")
        os.makedirs(subdir, exist_ok=True)
        
        src_image = f"tmp_motivation/output{id_}.jpg"
        dst_image = os.path.join(subdir, f"output{id_}.jpg")
        
        if os.path.exists(src_image):
            shutil.copy2(src_image, dst_image)
    
    # Process "bad" (high scores, descending order)
    for rank, (id_, _) in enumerate(reversed(high_ids), start=1):
        subdir = os.path.join(bad_dir, f"{rank:04d}")
        os.makedirs(subdir, exist_ok=True)
        
        src_image = f"tmp_motivation/output{id_}.jpg"
        dst_image = os.path.join(subdir, f"output{id_}.jpg")
        
        if os.path.exists(src_image):
            shutil.copy2(src_image, dst_image)




def auroc(heatmap, gt_binary):
    """Area Under ROC Curve"""
    return roc_auc_score(
        gt_binary.flatten().cpu().numpy(), 
        heatmap.flatten().cpu().numpy()
    )

def boundary_f1(pred_binary, gt_binary, dilation=2):
    """Boundary F1-score"""
    # Extract boundaries using morphological gradient
    from scipy.ndimage import binary_dilation, binary_erosion
    
    pred_np = pred_binary.cpu().numpy()
    gt_np = gt_binary.cpu().numpy()
    
    # Get boundaries
    pred_boundary = binary_dilation(pred_np) & ~binary_erosion(pred_np)
    gt_boundary = binary_dilation(gt_np) & ~binary_erosion(gt_np)
    
    # Dilate boundaries for tolerance
    pred_boundary_dilated = binary_dilation(pred_boundary, iterations=dilation)
    gt_boundary_dilated = binary_dilation(gt_boundary, iterations=dilation)
    
    # Precision: pred boundary pixels near gt boundary
    precision = np.sum(pred_boundary & gt_boundary_dilated) / (np.sum(pred_boundary) + 1e-8)
    
    # Recall: gt boundary pixels near pred boundary
    recall = np.sum(gt_boundary & pred_boundary_dilated) / (np.sum(gt_boundary) + 1e-8)
    
    # F1
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return f1

def hausdorff_distance(pred_binary, gt_binary):
    """Hausdorff distance (lower is better)"""
    pred_np = pred_binary.cpu().numpy().astype(bool)
    gt_np = gt_binary.cpu().numpy().astype(bool)
    
    # Distance transforms
    dist_pred_to_gt = distance_transform_edt(~gt_np)
    dist_gt_to_pred = distance_transform_edt(~pred_np)
    
    # Hausdorff: max of (max distance from pred to gt, max distance from gt to pred)
    hd1 = dist_pred_to_gt[pred_np].max() if pred_np.any() else 0
    hd2 = dist_gt_to_pred[gt_np].max() if gt_np.any() else 0
    
    return max(hd1, hd2)


def average_precision(heatmap, gt_binary):
    """Average Precision - no threshold needed"""
    return average_precision_score(
        gt_binary.flatten().cpu().numpy(), 
        heatmap.flatten().cpu().numpy()
    )



def playground3(args):
    ROOT = "tmp_motivation"   # contains "good" and "bad"
    GRID_W, GRID_H = 5, 3    # change to (5,4) if you really want 20 images
    N = GRID_W * GRID_H

    def make_grid(parent, out_name):
        imgs = []

        subdirs = sorted(
            [d for d in os.listdir(parent) if d.isdigit()],
            key=lambda x: int(x)
        )

        for d in subdirs:
            subdir = os.path.join(parent, d)
            files = [f for f in os.listdir(subdir)
                    if os.path.isfile(os.path.join(subdir, f))]

            if len(files) == 1:  # exactly one file → take it
                img_path = os.path.join(subdir, files[0])
                try:
                    imgs.append(Image.open(img_path))
                except Exception:
                    pass

            if len(imgs) == N:
                break

        if not imgs:
            return

        w, h = imgs[0].size
        grid = Image.new("RGB", (GRID_W * w, GRID_H * h))

        for i, img in enumerate(imgs):
            x = (i % GRID_W) * w
            y = (i // GRID_W) * h
            grid.paste(img, (x, y))

        grid.save(out_name)

    make_grid(os.path.join(ROOT, "good"), "good_grid.jpg")
    make_grid(os.path.join(ROOT, "bad"),  "bad_grid.jpg")




def playground2(args):
    #reorganize_samples()
    #exit(1)
    ROOT = Path("tmp_motivation") #tmp/motivation
    curves = []
    plt.figure(figsize=(8, 5))
    plt.rcParams.update({'font.size': 14}) 

    colors = ["blue", "orange", "red", "yellow"]
    count = 0
    mapper = {"good": "Plausible", "bad": "Implausible"}
    for subdir in ["good", "bad"]:
        for key in ["score"]: #  "score_aux"
           
            ts, mu, lo, hi = load_mean_and_ci(ROOT / subdir / "res.json", key)
            ts = [(50-i) for i in range(0,50,5)]
            plt.plot(ts, mu, color=colors[count], label=f"{mapper[subdir]}")
            plt.fill_between(ts, lo, hi, color=colors[count], alpha=0.15)
            plt.plot(ts, hi, color=colors[count], linewidth=0.4)  # top edge
            plt.plot(ts, lo, color=colors[count], linewidth=0.4)  # bottom edge
            count+=1


    # ---- Plot ----
    

    

    plt.xlabel("Timestep")
    plt.title("Mean Pixel-Wise Uncertainty")
    plt.legend()
    plt.grid(True)

    plt.xticks(ts)  # show only the timesteps you plotted
    plt.gca().invert_xaxis()  # optional: reverse axis like denoising steps

    plt.savefig("finalgraphNEW", dpi=300)
    exit(1)
    '''
    ROOT = Path("tmp_motivation")
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    colors = ["blue", "lightblue", "red", "lightcoral"]
    mapper = {"good": "Plausible", "bad": "Implausible"}
    
    # Create second y-axis
    ax2 = ax1.twinx()
    
    count = 0
    for subdir in ["good", "bad"]:
        # Plot "score" on left y-axis (ax1)
        ts, mu, lo, hi = load_mean_and_ci(ROOT / subdir / "res.json", "score")
        ax1.plot(ts, mu, color=colors[count], label=f"{mapper[subdir]}; Gradient-based", linestyle='-')
        ax1.fill_between(ts, lo, hi, color=colors[count], alpha=0.15)
        ax1.plot(ts, hi, color=colors[count], linewidth=0.4)
        ax1.plot(ts, lo, color=colors[count], linewidth=0.4)
        count += 1
        
        # Plot "score_aux" on right y-axis (ax2)
        ts, mu, lo, hi = load_mean_and_ci(ROOT / subdir / "res.json", "score_aux")
        ax2.plot(ts, mu, color=colors[count], label=f"{mapper[subdir]}; Divergence", linestyle='--')
        #ax2.fill_between(ts, lo, hi, color=colors[count], alpha=0.15)
        #ax2.plot(ts, hi, color=colors[count], linewidth=0.4)
        #ax2.plot(ts, lo, color=colors[count], linewidth=0.4)
        count += 1
    
    # Configure axes
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("Gradient-based", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True)
    
    ax2.set_ylabel("Divergence", color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Title
    plt.title("Mean Magnitude")
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    y1_min, y1_max = ax1.get_ylim()
    ax1.set_ylim(y1_min, y1_max * 1.15)  # extend top by 15%
   
    y2_min, y2_max = ax2.get_ylim()
    ax2.set_ylim(y2_min, y2_max * 1.15)  # extend top by 15%
    ax1.set_xticks(ts)
    ax1.invert_xaxis()
    
    plt.savefig("finalgraphNEW", dpi=300, bbox_inches='tight')
    exit(1)'''
    #all_metrics = plot_comparison_metrics(ROOT, method_names=["score", "score_aux"])


                        
                        


def playground4(args):
    #reorganize_samples()
    #exit(1)
    ROOT = Path("tmp_motivation_exp") #tmp/motivation
    curves = []
    plt.figure(figsize=(8, 5))
    plt.rcParams.update({'font.size': 14}) 

    colors = ["blue", "blue", "red", "red"]
    count = 0
    mapper = {"intersection_uncertainty_mean": "Recall; Gradients", "intersection_latent_mean": "Recall; Divergence","iou_uncertainty_mean": "IoU; Gradients", "iou_uncertainty_percentile": "Gradients" ,"iou_latent_mean": "IoU; Divergence", "iou_latent_percentile": "Divergence" }
  
    for key in ["intersection_uncertainty_mean", "iou_uncertainty_mean" , "intersection_latent_mean","iou_latent_mean" ]: #   "iou_uncertainty_mean", "iou_latent_mean"
        
        ts, mu, lo, hi = load_mean_and_ci(ROOT / "res.json", key)
        
        if count % 2 == 1:
            plt.plot(ts, mu, color=colors[count], label=f"{mapper[key]}", linestyle='--')
        else:
            plt.plot(ts, mu, color=colors[count], label=f"{mapper[key]}")
        #plt.fill_between(ts, lo, hi, color=colors[count], alpha=0.15)
        #plt.plot(ts, hi, color=colors[count], linewidth=0.4)  # top edge
        #plt.plot(ts, lo, color=colors[count], linewidth=0.4)  # bottom edge
        count+=1


    # ---- Plot ----
    

    

    plt.xlabel("Timestep")
    plt.title("IoU & Recall")
    plt.legend(loc="upper left")
    plt.grid(True)
    ymin, ymax = plt.ylim()
    plt.ylim(ymin, ymax * 1.5)
    plt.xticks(ts)  # show only the timesteps you plotted
    plt.gca().invert_xaxis()  # optional: reverse axis like denoising steps
    plt.margins(x=0)

    plt.savefig("finalgraphIOU", dpi=300)
    exit(1)
   

        
    

def split_lists(A, B):
    assert len(A) == len(B)

    n = len(A)
    half = n // 2

    # sort indices by A values (ascending)
    sorted_idx = sorted(range(n), key=lambda i: A[i])

    # split indices
    low_idx = sorted_idx[:half]            # lowest values
    high_idx = sorted_idx[half:]            # highest values

    # build lists
    A_low  = [A[i] for i in low_idx]         # ascending
    B_low  = [B[i] for i in low_idx]

    A_high = [A[i] for i in reversed(high_idx)]  # descending
    B_high = [B[i] for i in reversed(high_idx)]

    return A_low, A_high, B_low,  B_high



    '''import matplotlib as mpl

    cmap = plt.get_cmap("hot")  # or your custom 'chot'

    # Create figure with A4 width (landscape) and small height
    fig, ax = plt.subplots(figsize=(11.7, 0.6))  # width in inches, height small for colorbar

    # Create horizontal colorbar only
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='horizontal')

    # Remove default ticks
    cb.set_ticks([])

    # Add custom labels
    cb.ax.text(0.02, 0.5, "low", color='white', weight='bold', ha='left', va='center', fontsize=12)
    cb.ax.text(0.98, 0.5, "high", color='black', weight='bold', ha='right', va='center', fontsize=12)

    plt.tight_layout()

    # Save as image (PNG or any format)
    fig.savefig("colorbar_hot.png", dpi=300, bbox_inches='tight', transparent=False)

    plt.close(fig)  # close figure to free memory'''
    '''import matplotlib.pyplot as plt
 
    subdirs = sorted([d for d in os.listdir(args.output_dir) if os.path.isdir(os.path.join(args.output_dir, d))], 
                    key=lambda x: int(x))
    

    segments = range(12)
    for i, seg in enumerate(segments):
        print(f"{i}/12")
        sample_score_mapper = {}
        method = f"VARUC_{seg}"
        
        final_output_dir = f"{args.output_dir_compare}/{method}"
       

        print(final_output_dir)
        for idx, subdir in enumerate(subdirs):

            subdir_path = os.path.join(args.output_dir, subdir)
            
            with open(f"{subdir_path}/var_uc.json", "r") as f:
                data = json.load(f)
            
            sample_score_mapper[idx] = data[str(seg)]
        
        #print(sample_score_mapper)
        values = [sample_score_mapper[k] for k in sample_score_mapper]
    
        mean_val = np.mean(values)
       
        std_val  = np.std(values)
        
        

        
        # Plot
        plt.hist(values, bins='auto',density=True,  edgecolor='black', linewidth=0.5)
        plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=r"$\mu$")

        #plt.axvline(mean_val + std_val, color='blue', linestyle='--', linewidth=1.5, label=f"μ + σ = {mean_val+std_val:.6f}")
        #plt.axvline(mean_val - std_val, color='green', linestyle='--', linewidth=1.5, label=f"μ - σ = {mean_val-std_val:.6f}")

        plt.xticks(
            [mean_val - std_val, mean_val, mean_val + std_val],
            [r"$\mu - \sigma$", r"$\mu$", r"$\mu + \sigma$"]
        )

        #plt.xlabel("Value (μ = mean)")  # shows μ on the x-axis label
        plt.ylabel("Density")
        plt.title("Value Distribution")
        #plt.legend()
        plt.savefig(f"tmp/seg{i}.jpg", dpi=300, bbox_inches="tight")
        plt.show()
        plt.cla()'''

    '''with open(f"{args.output_vis_dir_compare}best_worst/res.json") as f:
        data = json.load(f)
    
    save_dir = f"{args.output_vis_dir_compare}best_worst/subset"
    os.makedirs(save_dir, exist_ok=True )
    for k in data:
        if "globalTimestep" not in k:
            continue
        res = data[k]
  
        for subdir in res:
          
            subdir_path = os.path.join(args.output_dir, subdir)
      
            src = Path(subdir_path)
            dst_root = Path(save_dir)
            dst = dst_root / src.name 
            shutil.copytree(src, dst, dirs_exist_ok=True)
    '''
    ''' subdirs = sorted([d for d in os.listdir(args.output_dir) if os.path.isdir(os.path.join(args.output_dir, d))], 
                        key=lambda x: int(x))
        for subdir in subdirs:
            subdir_path = os.path.join(args.output_dir, subdir)
            with open(f"{subdir_path}/prompt.txt", "r") as f:
                line = f.readline().strip()
            if "bull" in line:
                print(f"{subdir}:{line}")
                print("\n")'''
        

      