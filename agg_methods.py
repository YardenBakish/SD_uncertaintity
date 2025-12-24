import torch
#from diffusers import StableDiffusionPipeline

from modules.pipeline_stable_diffusion import StableDiffusionPipeline
from torchmetrics.multimodal import CLIPScore
from PIL import Image
from datasets import load_dataset
from modules.unet_2D_conditioned import UNet2DConditionModel
from modules.scheduling_pndm import PNDMScheduler
from torchvision import transforms
import os
import torch.nn.functional as F
from metrics import compute_fid_custom
from metrics2 import calculate_metrics
import matplotlib.pyplot as plt
from artifacts_heatmap_generator.RichHF.model import  preprocess_image, RAHF
import json
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils import *

from scipy.ndimage import (
    binary_erosion,
    gaussian_filter,
    label,
)


def compute_frame_differences(image_sequence):
    differences = []
    for i in range(1, len(image_sequence)):
        diff = np.abs(
            image_sequence[i].astype(np.float32)
            - image_sequence[i - 1].astype(np.float32)
        )
        differences.append(diff)
    return differences


def get_artifact_mask(
    image_sequence,
    mad_scale: float = 3,
    min_area: int = 100,
    max_area: int = 10000,
    min_width: int = 5,
    expand_size: int = 0,
) -> np.ndarray:
    smoothed_diffs = []
    
 
    for i in range(1, len(image_sequence)):
        diff = np.abs(
            image_sequence[i].astype(np.float32)
            - image_sequence[i - 1].astype(np.float32)
        )
        smoothed_diffs.append(gaussian_filter(diff, sigma=2))

    artifact_shape = (
        smoothed_diffs[0].shape
        if smoothed_diffs[0].ndim == 2
        else smoothed_diffs[0].shape[:2]
    )

    artifact_mask = np.zeros(artifact_shape, dtype=bool)

    for diff in smoothed_diffs:
        median = np.median(diff)
        mad = np.median(np.abs(diff - median))
        threshold = median + mad_scale * 1.4826 * mad
        
        diff_max = np.max(diff, axis=2) if diff.ndim == 3 else diff
       
        artifact_mask |= diff_max > threshold


    labeled_mask, num_features = label(artifact_mask)

   
    if num_features > 0:
        sizes = np.bincount(labeled_mask.ravel())[1:]
        mask_sizes = np.zeros_like(labeled_mask, dtype=bool)
        for i, size in enumerate(sizes, start=1):
            if min_area < size < max_area:
                mask_sizes[labeled_mask == i] = True
        artifact_mask = mask_sizes

    labeled_mask, num_features = label(artifact_mask)
    filtered_mask = np.zeros_like(artifact_mask, dtype=bool)
    structuring_element = np.ones((min_width, min_width), dtype=bool)

    for region_idx in range(1, num_features + 1):
        region = labeled_mask == region_idx
        eroded_region = binary_erosion(region, structure=structuring_element)
        if eroded_region.sum() > 0:
            filtered_mask |= region

    if expand_size > 0:
        mask_tensor = torch.from_numpy(filtered_mask).bool().float()
        kernel_size = 2 * expand_size + 1
        kernel = torch.ones(
            (1, 1, kernel_size, kernel_size), device=mask_tensor.device
        )
        expanded_mask = F.conv2d(
            mask_tensor.unsqueeze(0).unsqueeze(0), kernel, padding=expand_size
        )
        expanded_mask = (expanded_mask > 0).squeeze().numpy()
        return expanded_mask

    return filtered_mask







def otsu_threshold(img):
    """
    Compute Otsu's threshold for a 2D array.
    """
    # Flatten the image into 1D array
    flat = img.flatten()
    
    # Get histogram
    hist, bins = np.histogram(flat, bins=256, range=(0,1))
    hist = hist.astype(float)
    
    # Get bin centers
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # Get total number of pixels
    total = hist.sum()
    
    best_thresh = 0
    best_variance = 0
    
    # Calculate cumulative sums
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    
    # Calculate cumulative means
    mean1 = np.cumsum(hist * bin_centers) / weight1
    mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]
    
    # Calculate between class variance
    variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    
    # Get threshold with maximum variance
    idx = np.argmax(variance)
    best_thresh = bin_centers[idx]
    
    return best_thresh




 
def generate_single_map(latents_sample, agg_type, timestep_index):
    latent = latents_sample[timestep_index]
   
    if agg_type == "sum":
        return latent.sum(), None
    elif agg_type == "max":
        return latent.max(), None
    elif agg_type == "aboveAvg":
        latent = (latent - latent.min()) / (latent.max() - latent.min())
        bin_map =  latent > latent.mean()
        res = (bin_map).sum()
        return res, bin_map
    elif agg_type == "aboveOtsu":
        latent = (latent - latent.min()) / (latent.max() - latent.min())
        thr = otsu_threshold(latent.cpu().detach().numpy())
        bin_map = latent > thr
        return (bin_map).sum(), bin_map



def generate_single_map_global_method(latents_sample, agg_type):
    
    parts = agg_type.split('$')
    reduce_op, select_op = parts[0], parts[1]
    
    if 'Over' in reduce_op:
        # Aggregate across all maps
        if reduce_op == 'sumOver':
            result_map = torch.stack(latents_sample).sum(dim=0)
        else:  # maxOver
            result_map = torch.stack(latents_sample).max(dim=0)[0]
        
        score = result_map.sum().item()
        return score, result_map
    
    else:
        # Reduce each map to scalar, then select
        if reduce_op == 'sumEach':
            scores = [m.sum().item() for m in latents_sample]
        else:  # maxEach
            scores = [m.max().item() for m in latents_sample]
        
        best_idx = scores.index(max(scores))
        best_map = latents_sample[best_idx]
        
        if select_op == 'max':
            return scores[best_idx], best_map
        else:  # otsu
            threshold = otsu_threshold(best_map)  # Your external function
            score = (best_map > threshold).sum().item()
            return score, best_map



def generate_map_wrapper(x, method, methods_dict, dirs_dict, 
                        compare_mode=False, 
                        resize_fid=None, 
                        vis = False,
                        start_timestep = None,
                        end_timestep = None,

                        ):

    output_dir = dirs_dict["output_dir_compare"]
    final_output_dir = f"{output_dir}/{method}"
    real_dataset_dir = dirs_dict["real_dataset_dir"]
    fake_dataset_dir = dirs_dict["fake_dataset_dir"]

    #print(final_output_dir)
    #exit(1)

    all_unmaps, all_latents, time_steps_sorted =  x

    if end_timestep:
        time_steps_sorted = time_steps_sorted[:end_timestep]
        all_unmaps = [e[:end_timestep] for e in all_unmaps]
        all_latents = [e[:end_timestep] for e in all_latents]

    if start_timestep:
        time_steps_sorted = time_steps_sorted[start_timestep:]
        all_unmaps = [e[start_timestep:] for e in all_unmaps]
        all_latents = [e[start_timestep:] for e in all_latents]


    method_sep = method.split("_")
    agg_type = method_sep[1]
    type_method = method_sep[0]
    reps = all_unmaps if "latent" in method else all_latents

    

    uncertaintity_maps_dict = {}
    uncertaintity_maps = {}
    uncertaintity_maps_bin = {}

    for sample_idx, latents_sample in enumerate(reps): # 300000 times
        print(f"{sample_idx}/ {len(reps)}")
        reps_loaded = [torch.load(filepath) for filepath in latents_sample]
        if type_method == "basic" and "perTimestep" in method:
            timestep = int(method_sep[-1])
            timestep_index = time_steps_sorted.index(timestep)
            latent_sample_timestep  = reps_loaded[timestep_index]
            uncertaintity_maps[sample_idx] = latent_sample_timestep
            
            tmp_res = generate_single_map(reps_loaded, agg_type, timestep_index)

            uncertaintity_maps_dict[sample_idx] = tmp_res[0]  #score 
            if vis:
                uncertaintity_maps_bin[sample_idx] = tmp_res[1]
        
        elif type_method == "basic" and "globalTimestep" in method:
            tmp_res = generate_single_map_global_method(reps_loaded, agg_type)
            uncertaintity_maps_dict[sample_idx] = tmp_res[0]  #score 
            if vis:
                uncertaintity_maps[sample_idx] = tmp_res[1]

             
            
    d = {}
    if compare_mode == "fid_filter_high":
        file_ids = sorted(uncertaintity_maps_dict.items(), key=lambda x: x[1], reverse=True)
        file_ids = [elem[0] for elem in file_ids]
        len_file_ids = len(file_ids)
        file_ids_84 = file_ids[int(0.16 * len_file_ids ):]
      
        fid_res = compute_fid_custom(fake_dataset_dir, real_dataset_dir, file_indices = file_ids_84)
        

        d["fid"] = fid_res
        update_json(f"{final_output_dir}/res.json", d)
        
        exit(1)
        
      
        '''prec_rec_res = calculate_metrics(
            real_folder=real_dataset_dir,
            gen_folder=fake_dataset_dir,
            nhood_size=3,
            batch_size=32,
            file_indices = file_ids_84
        )

        precision = prec_rec_res["precision"]
        recall = prec_rec_res["recall"]'''


    return {'uncertaintity_maps_dict': uncertaintity_maps_dict,
            'uncertaintity_maps': uncertaintity_maps,
            'uncertaintity_maps_bin': uncertaintity_maps_bin
            }

        
        
        
    
    
    
    
    '''
    for k, v in sorted(uncertaintity_maps.items(), key=lambda x: x[1], reverse=True):
         print(f"{k}: {v}")

    print(output_dir)
    exit(1)'''




 
   








