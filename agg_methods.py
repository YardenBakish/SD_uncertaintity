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

import matplotlib.pyplot as plt
from artifacts_heatmap_generator.RichHF.model import  preprocess_image, RAHF

from torch.utils.data import Dataset, DataLoader
import numpy as np

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



def score_map(binary_map, value_map, score_type):
    if score_type == "sum":
        return value_map[binary_map].sum()
    elif score_type == "count":
        return binary_map.sum()
    else:
        raise ValueError(f"Unknown score_type: {score_type}")
    


local_methods = {
    "single_t_10": lambda x: local_single_timestep(x, t=10),
    "sum_all": lambda x: local_aggregate_all(x, agg="sum"),
    "max_all": lambda x: local_aggregate_all(x, agg="max"),
}


def local_aggregate_all(latent_sample, agg="sum"):
    stack = np.stack(latent_sample, axis=0)  # [T,H,W]
    if agg == "sum":
        return stack.sum(axis=0)
    elif agg == "max":
        return stack.max(axis=0)
    else:
        raise ValueError

 

def generate_map_single_step(latents, method, methods_dict):

    all_unmaps, all_latents =  x

    
    for idx, latent in enumerate(latents):
        latent_step_t =  latent[step]



