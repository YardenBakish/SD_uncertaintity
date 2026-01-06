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
        if "above" in method or ("count" in method and "_pr" not in method):
            continue
        #if "ASCED" not in method:
        #    continue
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


def playground(args):

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
    subdirs = sorted([d for d in os.listdir(args.output_dir) if os.path.isdir(os.path.join(args.output_dir, d))], 
                    key=lambda x: int(x))
    for subdir in subdirs:
        subdir_path = os.path.join(args.output_dir, subdir)
        with open(f"{subdir_path}/prompt.txt", "r") as f:
            line = f.readline().strip()
        if "bull" in line:
            print(f"{subdir}:{line}")
            print("\n")
        

      