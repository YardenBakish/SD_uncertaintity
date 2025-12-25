import torch
from torch.utils.data import Dataset, DataLoader

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
    
    print(image_sequence.shape)
    exit(1)
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

    

    return filtered_mask




class LatentFileDataset(Dataset):
    """Dataset that loads latent files on-demand."""
    
    def __init__(self, file_paths_list, start_idx=None, end_idx=None, asced = False):
        """
        Args:
            file_paths_list: List of lists, where each inner list contains file paths for one sample
            start_idx: Start timestep index (inclusive)
            end_idx: End timestep index (exclusive)
        """
        self.file_paths_list = file_paths_list
        self.start_idx = start_idx
        self.end_idx = end_idx

        self.asced = asced
        
        # Apply timestep filtering to paths
        if end_idx is not None:
            self.file_paths_list = [paths[:end_idx] for paths in file_paths_list]
        if start_idx is not None:
            self.file_paths_list = [paths[start_idx:] for paths in file_paths_list]
    
    def __len__(self):
        return len(self.file_paths_list)
    
    def __getitem__(self, idx):
        """Load all timesteps for a single sample."""
        file_paths = self.file_paths_list[idx]
        
        # Load all timesteps for this sample and stack into tensor
        # Shape: [num_timesteps, ...]
        
        if self.asced:
            tensors = torch.stack([torch.load(fp, map_location="cpu") for fp in file_paths])
           
        else:
            tensors = torch.stack([torch.load(fp) for fp in file_paths])
        
        return idx, tensors



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


def collate_fn(batch):
    """Custom collate to handle batched tensors."""
    indices = torch.tensor([item[0] for item in batch])
    # Stack all samples: [batch_size, num_timesteps, ...]
    tensors_batch = torch.stack([item[1] for item in batch])
    return indices, tensors_batch


def generate_batch_map_perTimestep(latents_batch, agg_type, timestep_index):
    """
    Process a batch of samples for perTimestep method.
    
    Args:
        latents_batch: [batch_size, num_timesteps, ...] tensor
        agg_type: Aggregation type
        timestep_index: Which timestep to use
    
    Returns:
        scores: [batch_size] tensor of scores
        bin_maps: [batch_size, ...] tensor of binary maps (or None)
    """
    # Select the specific timestep: [batch_size, ...]
    latent = latents_batch[:, timestep_index]
    
    if agg_type == "sum":
        # Sum over all dims except batch: [batch_size]
        scores = latent.flatten(1).sum(dim=1)
        return scores, None
    
    elif agg_type == "max":
        # Max over all dims except batch: [batch_size]
        scores = latent.flatten(1).max(dim=1)[0]
        return scores, None
    
    elif agg_type == "aboveAvg":
        # Normalize per sample
        batch_size = latent.shape[0]
        latent_flat = latent.view(batch_size, -1)
        
        mins = latent_flat.min(dim=1, keepdim=True)[0]
        maxs = latent_flat.max(dim=1, keepdim=True)[0]
        latent_norm = (latent_flat - mins) / (maxs - mins + 1e-8)
        
        means = latent_norm.mean(dim=1, keepdim=True)
        bin_map = latent_norm > means
        scores = bin_map.sum(dim=1)
        
        # Reshape back to original shape (except batch dim)
        bin_map = bin_map.view_as(latent)
        return scores, bin_map
    
    elif agg_type == "aboveOtsu":
        # For Otsu, we need to process each sample individually
        # since otsu_threshold is not vectorized
        scores = []
        bin_maps = []
        
        for i in range(latent.shape[0]):
            single_latent = latent[i]
            single_latent = (single_latent - single_latent.min()) / (single_latent.max() - single_latent.min() + 1e-8)
            thr = otsu_threshold(single_latent.cpu().detach().numpy())
            bin_map = single_latent > thr
            scores.append(bin_map.sum())
            bin_maps.append(bin_map)
        
        return torch.stack(scores), torch.stack(bin_maps)


def generate_batch_map_globalTimestep(latents_batch, agg_type):
    """
    Process a batch of samples for globalTimestep method.
    
    Args:
        latents_batch: [batch_size, num_timesteps, ...] tensor
        agg_type: Aggregation type (e.g., "sumOver$max")
    
    Returns:
        scores: [batch_size] tensor of scores
        result_maps: [batch_size, ...] tensor of maps
    """
    parts = agg_type.split('$')
    reduce_op, select_op = parts[0], parts[1]
    
    if 'Over' in reduce_op:
        # Aggregate across timesteps: [batch_size, ...]
        if reduce_op == 'sumOver':
            result_maps = latents_batch.sum(dim=1)
        else:  # maxOver
            result_maps = latents_batch.max(dim=1)[0]
        
        # Sum over spatial dims: [batch_size]
        scores = result_maps.flatten(1).sum(dim=1)
        return scores, result_maps
    
    else:
        # Reduce each timestep to scalar, then select best
        batch_size = latents_batch.shape[0]
        
        if reduce_op == 'sumEach':
            # [batch_size, num_timesteps]
            scores_per_timestep = latents_batch.flatten(2).sum(dim=2)
        else:  # maxEach
            # [batch_size, num_timesteps]
            scores_per_timestep = latents_batch.flatten(2).max(dim=2)[0]
        
        # Find best timestep per sample
        best_indices = scores_per_timestep.argmax(dim=1)  # [batch_size]
        
        # Select best map for each sample
        batch_indices = torch.arange(batch_size)
        best_maps = latents_batch[batch_indices, best_indices]  # [batch_size, ...]
        
        if select_op == 'max':
            best_scores = scores_per_timestep[batch_indices, best_indices]
            return best_scores, best_maps
        else:  # otsu
            # Process each sample individually for Otsu
            scores = []
            for i in range(batch_size):
                threshold = otsu_threshold(best_maps[i].cpu().detach().numpy())
                score = (best_maps[i] > threshold).sum()
                scores.append(score)
            return torch.stack(scores), best_maps




def generate_map_wrapper(x, method, methods_dict, dirs_dict, 
                        compare_mode=False, 
                        resize_fid=None, 
                        vis=False,
                        start_timestep=None,
                        end_timestep=None,
                        batch_size=32,
                        asced = False,
                        mad_value = None,
                        num_workers=4):
    """
    Optimized version using DataLoader for parallel loading + batch processing.
    
    Args:
        batch_size: Number of samples to process at once
        num_workers: Number of parallel workers for data loading
    """
    output_dir = dirs_dict["output_dir_compare"]
    final_output_dir = f"{output_dir}/{method}"
    real_dataset_dir = dirs_dict["real_dataset_dir"]
    fake_dataset_dir = dirs_dict["fake_dataset_dir"]

    
    all_unmaps, all_latents, time_steps_sorted = x

    # Apply timestep filtering to time_steps_sorted
    filtered_time_steps = time_steps_sorted
 

    method_sep = method.split("_")
    agg_type = method_sep[1]
    type_method = method_sep[0]
    reps = all_latents  if "Latent" in method else all_unmaps

    

    # Create dataset and dataloader
    dataset = LatentFileDataset(reps, start_timestep, end_timestep, asced = asced)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True,
        #prefetch_factor=2
    )

    uncertaintity_maps_dict = {}
    uncertaintity_maps = {}
    uncertaintity_maps_bin = {}

    # Process batches
    print(f"Processing {len(dataset)} samples in batches of {batch_size}...")
    
    for batch_idx, (indices, latents_batch) in enumerate(dataloader):

        

        if batch_idx % 10 == 0:
            processed = batch_idx * batch_size
            print(f"Batch {batch_idx}/{len(dataloader)} ({processed}/{len(dataset)} samples)")
        
        # Process entire batch at once
        if type_method == "basic" and "perTimestep" in method:
            timestep = int(method_sep[-1])
            timestep_index = filtered_time_steps.index(timestep)
            
            scores, bin_maps = generate_batch_map_perTimestep(latents_batch, agg_type, timestep_index)
            
            # Store results
            for i, sample_idx in enumerate(indices.tolist()):
                uncertaintity_maps_dict[sample_idx] = scores[i].item()
                uncertaintity_maps[sample_idx] = latents_batch[i, timestep_index]
                if vis and bin_maps is not None:
                    uncertaintity_maps_bin[sample_idx] = bin_maps[i]
        
        elif type_method == "basic" and "globalTimestep" in method:
            scores, result_maps = generate_batch_map_globalTimestep(latents_batch, agg_type)
            
            # Store results
            for i, sample_idx in enumerate(indices.tolist()):
                uncertaintity_maps_dict[sample_idx] = scores[i].item()
                if vis:
                    uncertaintity_maps[sample_idx] = result_maps[i]
        elif  type_method == "ASCED":
            
            artifact_mask = get_artifact_mask(
                latents_batch,
                mad_scale= mad_value,
                min_area = 100,
                max_area =  10000,
                min_width = 5,
                expand_size = 0,
            )

    d = {}
    if compare_mode == "fid_filter_high":
        file_ids = sorted(uncertaintity_maps_dict.items(), key=lambda x: x[1], reverse=True)
        file_ids = [elem[0] for elem in file_ids]
        len_file_ids = len(file_ids)
        file_ids_84 = file_ids[int(0.16 * len_file_ids):]

        prec_rec_res = calculate_metrics(
            real_folder=real_dataset_dir,
            gen_folder=fake_dataset_dir,
            nhood_size=3,
            batch_size=32,
            file_indices = file_ids_84
        )

        d["precision"] = prec_rec_res["precision"]
        d["recall"] = prec_rec_res["recall"]
      
        fid_res = compute_fid_custom(fake_dataset_dir, real_dataset_dir, file_indices=file_ids_84)
        d["fid"] = fid_res
        
        update_json(f"{final_output_dir}/res.json", d)

        
        
        
    
    return {'uncertaintity_maps_dict': uncertaintity_maps_dict,
            'uncertaintity_maps': uncertaintity_maps,
            'uncertaintity_maps_bin': uncertaintity_maps_bin
            }