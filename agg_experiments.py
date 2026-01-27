import torch
from torch.utils.data import Dataset, DataLoader

import torch
#from diffusers import StableDiffusionPipeline
from diffusers import DDIMScheduler
from official_supervised_metrics import run_sup_metrics
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
from metrics_clipscore import calculate_clipscore_metric
from metrics_cmmd import mainFunc
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
    return_map: bool=False,
    agg_type = None,
) -> np.ndarray:
    smoothed_diffs = []
    
    image_sequence = image_sequence.permute(0, 1, 3, 4, 2).cpu().numpy()
    B, N, H, W, C = image_sequence.shape

    # Vectorized diff calculation: [B, N-1, 64, 64, 4]
    diffs = np.abs(
        image_sequence[:, 1:].astype(np.float32) - 
        image_sequence[:, :-1].astype(np.float32)
    )

    # Apply Gaussian smoothing to all diffs at once
    # Reshape to apply filter: [B*(N-1), 64, 64, 4]
    smoothed_diffs = np.zeros_like(diffs)
    for b in range(B):
        for i in range(N - 1):
            smoothed_diffs[b, i] = gaussian_filter(diffs[b, i], sigma=2)

    artifact_masks = np.zeros((B, H, W), dtype=bool)
    diff_max = np.max(smoothed_diffs, axis=-1)

    
    # Reshape to [B, N-1, H*W*C] for median calculations
    flat_diffs = smoothed_diffs.reshape(B, N-1, -1)
    medians = np.median(flat_diffs, axis=2, keepdims=True) # [B, N-1, 1]
    mads = np.median(np.abs(flat_diffs - medians), axis=2, keepdims=True)  # [B, N-1, 1]
    thresholds = medians + mad_scale * 1.4826 * mads

    thresholds = thresholds.reshape(B, N-1, 1, 1)
    artifact_masks = np.any(diff_max > thresholds, axis=1)

    
    '''print(artifact_masks.shape)

    print(artifact_masks[0].sum())
    exit(1)'''

    if agg_type == "sum" or agg_type == "max":
        cond  = diff_max > thresholds
        if return_map:
            return (diff_max * cond).sum(axis=(1))
        scores = (diff_max * cond).sum(axis=(1,2,3))
        return scores  
        
   
    filtered_masks = np.zeros((B, H, W), dtype=bool)
    
    

    for b in range(B):
        labeled_mask, num_features = label(artifact_masks[b])


        if num_features > 0:
            sizes = np.bincount(labeled_mask.ravel())[1:]
            mask_sizes = np.zeros_like(labeled_mask, dtype=bool)
            for i, size in enumerate(sizes, start=1):
                if min_area < size < max_area:
                    mask_sizes[labeled_mask == i] = True
            artifact_masks[b] = mask_sizes

        labeled_mask, num_features = label(artifact_masks[b])
        filtered_masks[b] = np.zeros_like(artifact_masks[b], dtype=bool)
        structuring_element = np.ones((min_width, min_width), dtype=bool)

        for region_idx in range(1, num_features + 1):
            region = labeled_mask == region_idx
            eroded_region = binary_erosion(region, structure=structuring_element)
            if eroded_region.sum() > 0:
                filtered_masks[b] |= region
        
        
    return filtered_masks.sum(axis=(1, 2))
    #return filtered_masks.sum(axis=(1, 2))




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
            self.file_paths_list = [paths[:end_idx] for paths in self.file_paths_list]
        
        
        if start_idx is not None:
            self.file_paths_list = [paths[start_idx:] for paths in self.file_paths_list]
       
    
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




def generate_special_batch_map_globalTimestep(latents_batch, agg_type, model_path = None, global_ts = None):
    B, N, H, W = latents_batch.shape
    
    scheduler = DDIMScheduler.from_config(model_path, subfolder="scheduler",torch_dtype=torch.float16,)
    alphas_cumprod = scheduler.alphas_cumprod
    alpha_t_values = alphas_cumprod
    #print(alpha_t_values)
    #alpha_t_values[0] -=1
  
    weights = (1 - alpha_t_values) / torch.sqrt(alpha_t_values)

    weights = weights[np.array(global_ts)]
    
    use_weights = 'Weighted' in agg_type
    if use_weights:
        w = weights.view(1, N, 1, 1)
    accumulated = latents_batch[:, 0, :, :].clone()

    if 'diff' in agg_type:
        # Compute differences between consecutive maps
        # Shape: (B, N-1, H, W)
        diffs = latents_batch[:, 1:, :, :] - latents_batch[:, :-1, :, :]
        if use_weights:
            diffs = diffs * w[:, 1:, :, :]  # Apply weights[1:] to diffs
        # Accumulate differences to the first map
        # Sum along the sequence dimension (dim=1)
        accumulated = accumulated + diffs.sum(dim=1)
    elif 'pr' in agg_type:
        # Process each map in the sequence (starting from index 1)
        for i in range(N):
            current_map = latents_batch[:, i, :, :]
            
            # Compute 95th percentile threshold per batch
            # Flatten spatial dimensions for percentile computation
            # Shape: (B, H*W)
            flat_maps = current_map.reshape(B, -1)
            
            # Compute threshold per batch: Shape (B, 1)
            thresholds = torch.quantile(flat_maps.float(), 0.95, dim=1, keepdim=True)
            
            # Reshape threshold to broadcast: (B, 1, 1)
            thresholds = thresholds.unsqueeze(-1)
            
            # Create mask and filter
            mask = current_map > thresholds
            filtered = current_map * mask
            if use_weights:
                filtered = filtered * weights[i]
            # Add filtered map to accumulator
            accumulated = accumulated + filtered
    
    # Apply absolute value to accumulated maps
    accumulated = torch.abs(accumulated)
    if "MAX" in agg_type:
        result = accumulated.amax(dim=(1, 2))

    else:
        result = accumulated.sum(dim=(1, 2))

   
    return result, accumulated
    





def generate_batch_map_globalTimestep(latents_batch, agg_type, model_path = None, global_ts = None):
    """
    Process a batch of samples for globalTimestep method.
    
    Args:
        latents_batch: [batch_size, num_timesteps, ...] tensor
        agg_type: Aggregation type (e.g., "sumOver$max")
    
    Returns:
        scores: [batch_size] tensor of scores
        result_maps: [batch_size, ...] tensor of maps
    """

    if '$' not in agg_type:
        return generate_special_batch_map_globalTimestep(latents_batch, agg_type, model_path = model_path, global_ts = global_ts)


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
                        backup_best_worst = False,
                        num_workers=4,
                        calc_clipscore = False,
                        calc_cmmd = False,
                        calc_sup_metrics = False,
                        calc_grad_fid = False,
                        vis_score_dist = False):
    
    
    
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
    output_dir_vis = dirs_dict["compare_vis_dir"]
    
    model_name = output_dir_vis.split("/")[-1]
    model_path = "stabilityai/stable-diffusion-xl-base-1.0" if model_name == "SDXL" else "runwayml/stable-diffusion-v1-5"
   
    all_unmaps, all_latents, time_steps_sorted = x

    

    # Apply timestep filtering to time_steps_sorted
    filtered_time_steps = time_steps_sorted
    
    global_ts = None
    if end_timestep is not None:
        global_ts = filtered_time_steps[:end_timestep]
        
    if start_timestep is not None:
        global_ts = global_ts[start_timestep:] 
   
    method_sep = method.split("_")
    agg_type = method_sep[1]
    type_method = method_sep[0]
    reps = all_latents  if "Latent" in method else all_unmaps

    

    # Create dataset and dataloader
    dataset = LatentFileDataset(reps, start_timestep, end_timestep, asced = asced)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=2
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
                if backup_best_worst == False:
                    uncertaintity_maps[sample_idx] = latents_batch[i, timestep_index]
                    if vis and bin_maps is not None:
                        uncertaintity_maps_bin[sample_idx] = bin_maps[i]
        
        elif type_method == "basic" and "globalTimestep" in method:
            scores, result_maps = generate_batch_map_globalTimestep(latents_batch, agg_type, model_path, global_ts)
            
            # Store results
            for i, sample_idx in enumerate(indices.tolist()):
                uncertaintity_maps_dict[sample_idx] = scores[i].item()
                if vis and backup_best_worst == False:
                    uncertaintity_maps[sample_idx] = result_maps[i]
        elif  type_method == "ASCED":
            
            scores = get_artifact_mask(
                latents_batch,
                mad_scale= mad_value,
                min_area = 100,
                max_area =  10000,
                min_width = 5,
                expand_size = 0,
                agg_type = agg_type,
            )
            for i, sample_idx in enumerate(indices.tolist()):
                #print(scores[i].item())
                uncertaintity_maps_dict[sample_idx] = scores[i].item()
            

    d = {}
    if compare_mode == "fid_filter_high":

        if vis_score_dist:
            import matplotlib.pyplot as plt
            output_vis_dir = dirs_dict["compare_vis_dir"]
            values = list(uncertaintity_maps_dict.values())

            print(min(values))
            print(max(values))

            mean_val = np.mean(values)
            print(mean_val)
            std_val  = np.std(values)

            #threshold = mean_val + std_val
            #count_above = np.sum(values > threshold)
            
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
            plt.savefig(f"{output_vis_dir}/dist_{method}.jpg", dpi=300, bbox_inches="tight")
            plt.show()
            exit(1)
            
            return


        if calc_grad_fid:
            
            file_ids = sorted(uncertaintity_maps_dict.items(), key=lambda x: x[1], reverse=False)
            file_ids = [elem[0] for elem in file_ids]
            len_file_ids = len(file_ids)
     
            for i in range(5):
                start = i * len_file_ids // 5
                end = (i + 1) * len_file_ids // 5
                segment = file_ids[:start] + file_ids[end:]
                
                d[i] = compute_fid_custom(fake_dataset_dir, real_dataset_dir, file_indices=segment)

            
            update_json(f"tmp/res_{method}.json", d)
              
               

            
            return 
           

            
        file_ids = sorted(uncertaintity_maps_dict.items(), key=lambda x: x[1], reverse=True)
        file_ids = [elem[0] for elem in file_ids]
    
        
        len_file_ids = len(file_ids)
        file_ids_84 = file_ids[int(0.16 * len_file_ids):]

        if calc_sup_metrics:
            pick_score, hpsv2_score, aes_score = run_sup_metrics(fake_dataset_dir, file_indices = file_ids_84)
            d["pick_score"] = pick_score
            d["hpsv2_score"] = hpsv2_score 
            d["aes_score"] = aes_score
            update_json(f"{final_output_dir}/res.json", d)
            return

 
        if calc_cmmd:
            d["cmmd"] = mainFunc(real_dataset_dir,fake_dataset_dir,file_ids_84)
           
            d["cmmd"] = float(d["cmmd"])
            update_json(f"{final_output_dir}/res.json", d)
  
            return 


        if calc_clipscore:
            d["clipscore"] = calculate_clipscore_metric(fake_dataset_dir, batch_size=5, file_indices = file_ids_84)
            update_json(f"{final_output_dir}/res.json", d)
            return 
        
        fid_res = compute_fid_custom(fake_dataset_dir, real_dataset_dir, file_indices=file_ids_84)
        d["fid"] = fid_res

        prec_rec_res = calculate_metrics(
            real_folder=real_dataset_dir,
            gen_folder=fake_dataset_dir,
            nhood_size=3,
            batch_size=32,
            file_indices = file_ids_84
        )

        d["precision"] = prec_rec_res["precision"]
        d["recall"] = prec_rec_res["recall"]
      
        
        
        update_json(f"{final_output_dir}/res.json", d)
        

        
    if vis and backup_best_worst:
        best_worst_output_dir = f"{output_dir_vis}/best_worst"
     
        os.makedirs(best_worst_output_dir, exist_ok=True)
       
        sorted_samples = sorted(uncertaintity_maps_dict.items(), key=lambda x: x[1], reverse=True)
        sorted_samples = [elem[0] for elem in sorted_samples]
        best_worst_keys = sorted_samples[:100] + sorted_samples[-100:]
        uncertaintity_maps_dict = {k: uncertaintity_maps_dict[k] for k in best_worst_keys }

        uncertaintity_maps_copy = {}
        uncertaintity_maps_bin_copy = {}
        for k in sorted_samples:
            if k in uncertaintity_maps:
                uncertaintity_maps_copy[k] = uncertaintity_maps[k]
            if k in uncertaintity_maps_bin:
                uncertaintity_maps_bin_copy[k] = uncertaintity_maps_bin[k]
        
        uncertaintity_maps = uncertaintity_maps_copy
        uncertaintity_maps_bin = uncertaintity_maps_bin_copy
        
        update_json(f"{best_worst_output_dir}/res.json", {method : uncertaintity_maps_dict})
        



    return {'uncertaintity_maps_dict': uncertaintity_maps_dict,
            'uncertaintity_maps': uncertaintity_maps,
            'uncertaintity_maps_bin': uncertaintity_maps_bin
            }