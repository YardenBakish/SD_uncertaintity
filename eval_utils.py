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
import json
import matplotlib.pyplot as plt
from agg_methods import get_artifact_mask, compute_frame_differences
from agg_experiments import generate_map_wrapper
from artifacts_heatmap_generator.RichHF.model import  preprocess_image, RAHF
from diffusers import DDIMScheduler

from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
from scipy.ndimage import (
    binary_erosion,
    gaussian_filter,
    label,
)
from scipy.ndimage import binary_dilation
def add_circular_border(mask, border_width=2):
    dilated = binary_dilation(mask, iterations=border_width)
    border = dilated & ~mask
    return border


def visualize_artifacts(
    original_image, artifact_mask, alpha=0.5, border_width=2, border_color="green",
):
    import cv2

    original_image = np.asarray(original_image, dtype=np.uint8)
    if original_image.dtype != np.uint8:
        original_image = (original_image * 255).astype(np.uint8)
    
    
    artifact_mask = cv2.resize(artifact_mask.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST).astype(bool)
    original_image_permute = torch.from_numpy(original_image).permute(2, 0, 1).unsqueeze(0).float()

    original_image = torch.nn.functional.interpolate(
                    original_image_permute,
                    size=(512, 512),
                    mode='bilinear',
                    align_corners=False
                ).squeeze().permute(1, 2, 0).numpy().astype(np.uint8)

    red_fill = np.zeros_like(original_image)
    red_fill[artifact_mask] = [255, 0, 0]

    border_mask = add_circular_border(artifact_mask, border_width)

    green_border = np.zeros_like(original_image)
    green_border[border_mask] = [0, 255, 0] if border_color == "green" else border_color

    result = np.where(
        artifact_mask[:, :, None],
        original_image * (1 - alpha) + red_fill * alpha,
        original_image,
    )
    result = np.where(border_mask[:, :, None], green_border, result)

    return result.astype(np.uint8)


#maintains only last layer + removes cond 
#structure: {t_N: (B,F)}

def simplify_uncertainty_maps(uncertainty_maps):
    timesteps = sorted(uncertainty_maps.keys(), reverse=True)
    last_layer = len(uncertainty_maps[timesteps[-1]]) -1
    uncertainty_maps_last_layer = { k : uncertainty_maps[k][last_layer].chunk(2)[0] for k in uncertainty_maps}
    return uncertainty_maps_last_layer




def plot_ASCD(
    latents_lst, 
    images,
    prompts = None,
    uncertainty_maps = None,
    out_dir = "uncertaintity_maps",
    target_size=128,
    cmap="hot",
    start_idx = 5,
    dpi=150,
    ours = False,
    ):


    start_timestep = 12# 12
    end_timestep   = 32 if ours is False else 32
    os.makedirs(out_dir, exist_ok=True)

   
    uncertainty_maps = simplify_uncertainty_maps(uncertainty_maps)

    sequence = latents_lst if ours is False else uncertainty_maps

    
    timesteps = sorted(uncertainty_maps.keys(), reverse=True)
  
    sequence_len = latents_lst[0].shape[0] if ours is False else sequence[timesteps[0]].shape[0]

    for sample_id in range(sequence_len):
        if ours is False:
            image_sequence = [np.array(latent[sample_id].cpu()).transpose(1, 2, 0) for latent in sequence]
            
        else:
            image_sequence = [np.array(sequence[ts][sample_id].cpu()).transpose(1, 2, 0) for ts in sequence]

        #print(image_sequence[0].shape)
        #exit(1)
        image_sequence = image_sequence[start_timestep:end_timestep]

        differences = compute_frame_differences(image_sequence)
        artifact_mask = get_artifact_mask(
            image_sequence,
            mad_scale= 3 if ours is False else 10,
            min_area = 4,
            max_area = 5000,
            min_width = 1,
            expand_size = 0,
        )


        result_image = visualize_artifacts(images[sample_id], artifact_mask, border_width=2)
        img          = Image.fromarray(result_image.astype(np.uint8))
        img.save(f"{out_dir}/binary_mask{start_idx+sample_id}.png")

        artifact_data = []
        non_artifact_data = []

        for diff in differences:
            artifact_data.append(diff[artifact_mask])
            non_artifact_data.append(diff[~artifact_mask])


        artifact_acceleration = np.diff([np.mean(d) for d in artifact_data])
        non_artifact_acceleration = np.diff([np.mean(d) for d in non_artifact_data])

        fig = plt.figure(figsize=(28, 7.5), constrained_layout=True)
        gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.8])

        ax1 = fig.add_subplot(gs[0])
        result_image = visualize_artifacts(images[sample_id], artifact_mask, border_width=2, alpha=0.0, border_color=[255,0,0])
        ax1.imshow(result_image)
        ax1.set_title(f"Detected Artifacts", fontsize=14)
        ax1.axis("off")

        #print(len(differences))
        #exit(1)

        #differences = differences[::2]

        num_images = len(differences)
        cols = math.ceil(math.sqrt(num_images))
        rows = math.ceil(num_images / cols)
        
        inner_gs = gs[1].subgridspec(rows, cols, wspace=0.05, hspace=0.05)
        for i in range(num_images):
            sub_ax = fig.add_subplot(inner_gs[i])
            im = sub_ax.imshow(differences[i].mean(2), cmap='viridis')
            sub_ax.set_title(f"$\Delta_{{{i + start_timestep + 1}}}$", fontsize=14)
            sub_ax.axis('off')
            fig.colorbar(im, ax=sub_ax, fraction=0.046, pad=0.02)

        ax3 = fig.add_subplot(gs[2])
        ax3.plot(artifact_acceleration, label='Artifact region', color='red', linewidth=2)
        ax3.plot(non_artifact_acceleration, label='Non-artifact region', color='blue', linewidth=2)
        ax3.set_title('Acceleration Comparison', fontsize=14)
        ax3.set_xlabel('Time', fontsize=12)
        ax3.set_ylabel('Acceleration', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.savefig(f"{out_dir}/accelration_mask{start_idx+sample_id}.png", dpi=300, bbox_inches="tight")


       

        #for diff in differences:
        #    artifact_data.append(diff[artifact_mask])
        #    non_artifact_data.append(diff[~artifact_mask])

        
    
              


def prepare_culumative(num_samples, last_layer_idx, uncertainty_maps, timesteps):

    scheduler = DDIMScheduler.from_config("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler",torch_dtype=torch.float16,)
    alphas_cumprod = scheduler.alphas_cumprod
    #print(timesteps.int())
    #print(len(alphas_cumprod))
    
    alpha_t_values = alphas_cumprod
    alpha_t_values[0] -=1
    weights = (1 - alpha_t_values) / torch.sqrt(alpha_t_values)

    #weights = weights / weights.sum()
    #print(weights)
    #exit(1)
    start_idx = 12
    timesteps = timesteps[start_idx:]
    for sample_idx in range(num_samples):
        for map_idx in range(2):
            differences = []
            for idx, ts in enumerate(timesteps):
                if idx == len(timesteps) -5:
                    break
                uncertainty = uncertainty_maps[ts][last_layer_idx].chunk(2)[map_idx][sample_idx]
                uncertainty_next = uncertainty_maps[timesteps[idx+1]][last_layer_idx].chunk(2)[map_idx][sample_idx]
                uncertainty_maps[timesteps[0]][last_layer_idx].chunk(2)[map_idx][sample_idx] += (weights[ts] * (uncertainty_next - uncertainty))
            
            uncertainty_maps[timesteps[0]][last_layer_idx].chunk(2)[map_idx][sample_idx] = torch.abs(uncertainty_maps[timesteps[0]][last_layer_idx].chunk(2)[map_idx][sample_idx])
                
   



def plot_uncertintiy_maps(
    uncertainty_maps, 
    images,
    prompts = None,
    out_dir = "uncertaintity_maps",
    target_size=128,
    cmap="hot",
    start_idx = 0,
    culumative = False,
    dpi=150):

    os.makedirs(out_dir, exist_ok=True)
    timesteps = sorted(uncertainty_maps.keys(), reverse=True)

    
    n_timesteps = len(timesteps)
    example_ts = timesteps[0]

    # Number of samples
    n_cols = len(uncertainty_maps[example_ts])#uncertainty_maps[example_ts][0].shape[0] // 2
    n_cols = 1 + n_cols
    last_layer_idx = n_cols - 2  # Last uncertainty map column
   
    num_samples = len(images)

    # Image resize
    img_resize = transforms.Resize((target_size, target_size))

    if culumative:
        prepare_culumative(num_samples, last_layer_idx, uncertainty_maps, timesteps)

    for sample_idx in range(num_samples):
        if False: #vis by layer
            for map_idx in range(2):
            
                fig, axes = plt.subplots(
                    n_timesteps,
                    n_cols,
                    figsize=(4.5 * n_cols, 4.0 * n_timesteps)
                )

                # Handle single-row case
                if n_timesteps == 1:
                    axes = axes.reshape(1, -1)

                for row_idx, ts in enumerate(timesteps):
                    # ------------------
                    # Column 0: Image
                    # ------------------
                    ax = axes[row_idx, 0]

                    img = img_resize(images[sample_idx])
                    ax.imshow(img)
                    ax.set_ylabel(
                        f"t = {ts}",
                        rotation=0,
                        labelpad=40,
                        va="center",
                        fontsize=24
                    )

                    ax.set_xticks([])
                    ax.set_yticks([])
                    for spine in ax.spines.values():
                        spine.set_visible(False)


                    for col_idx in range(n_cols-1):
                        col = col_idx + 1

                        #print(row_idx, col)
                    
                        uncertainty = uncertainty_maps[ts][col_idx].chunk(2)[map_idx][sample_idx].squeeze(0)
                        if uncertainty.max().item() == uncertainty.min().item():
                            uncertainty*=0
                        else:
                            uncertainty = (uncertainty - uncertainty.min() ) / (uncertainty.max()- uncertainty.min())

                        


                        axes[row_idx, col].imshow(uncertainty, cmap='hot')
                        if row_idx == 0:
                            axes[row_idx, col].set_title(f'upConv{col}', fontsize=18, pad=12)
                        axes[row_idx, col].axis('off')
                        
                        
                plt.suptitle(f'Sample {sample_idx+start_idx}', 
                            fontsize=20, fontweight='bold',y=0.98)
                plt.tight_layout(rect=[0.08, 0.05, 1, 0.95])
                vis_file_name = f'sample_{sample_idx+start_idx}_uncond.png' if map_idx == 0 else f'sample_{sample_idx+start_idx}_cond.png'
                plt.savefig(os.path.join(out_dir, f'{vis_file_name}'), dpi=150, bbox_inches='tight')
                plt.close()


        # ------------------
        # Create enlarged visualizations for the last layer
        # ------------------
        enlarged_vis_dir = os.path.join(out_dir, 'full_review')
        os.makedirs(enlarged_vis_dir, exist_ok=True)
       
        if n_timesteps > 6:

            #import numpy as np
            for map_idx in range(2):
                # NEW VISUALIZATION: Grid layout for many timesteps
                # Calculate grid dimensions
                grid_cols = int(np.ceil(np.sqrt(n_timesteps + 1)))  # +1 for the image
                grid_rows = int(np.ceil((n_timesteps + 1) / grid_cols))
                
                fig = plt.figure(figsize=(4 * grid_cols, 4 * grid_rows))
                gs = fig.add_gridspec(grid_rows, grid_cols, hspace=0.4, wspace=0.3)
                
                # First cell: Original image resized to 64x64
                ax_img = fig.add_subplot(gs[0, 0])
                img_resized = torch.nn.functional.interpolate(
                    torch.from_numpy(np.array(images[sample_idx])).permute(2, 0, 1).unsqueeze(0).float(),
                    size=(256, 256),
                    mode='bilinear',
                    align_corners=False
                ).squeeze().permute(1, 2, 0).numpy().astype(np.uint8)
                ax_img.imshow(img_resized)
                ax_img.set_title('Original Image', fontsize=12, fontweight='bold')
                ax_img.set_xticks([])
                ax_img.set_yticks([])
                for spine in ax_img.spines.values():
                    spine.set_visible(False)
                
                # Remaining cells: Uncertainty maps with colorbars
                for idx, ts in enumerate(timesteps):
                    cell_idx = idx + 1  # Offset by 1 because first cell is the image
                    row = cell_idx // grid_cols
                    col = cell_idx % grid_cols
                    
                    ax_map = fig.add_subplot(gs[row, col])
                    uncertainty = uncertainty_maps[ts][last_layer_idx].chunk(2)[map_idx][sample_idx].squeeze(0)
                    
                    if uncertainty.max().item() == uncertainty.min().item():
                        uncertainty *= 0
                    
                    # Resize uncertainty map to 64x64
                    uncertainty_resized = torch.nn.functional.interpolate(
                        uncertainty.unsqueeze(0).unsqueeze(0),
                        size=(256, 256),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze()
                    
                    im = ax_map.imshow(uncertainty_resized, cmap='hot')
                    ax_map.set_title(f't = {ts}', fontsize=10, fontweight='bold')
                    ax_map.set_xticks([])
                    ax_map.set_yticks([])
                    for spine in ax_map.spines.values():
                        spine.set_visible(False)
                    
                    # Add colorbar next to each map
                    cbar = plt.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)
                    cbar.ax.tick_params(labelsize=8)
                
                map_type = 'uncond' if map_idx == 0 else 'cond'
                plt.suptitle(
                    f'Sample {sample_idx+start_idx} - {map_type.upper()} - Last Layer', 
                    fontsize=16, 
                    fontweight='bold'
                )

                enlarged_file_name = f'enlarged_sample_{sample_idx+start_idx}_{map_type}.png'
                plt.savefig(
                    os.path.join(enlarged_vis_dir, enlarged_file_name),
                    dpi=dpi,
                    bbox_inches='tight'
                )
                plt.close()
              
        else:
       
            enlarged_vis_dir = os.path.join(out_dir, 'enlarged_dir')
            os.makedirs(enlarged_vis_dir, exist_ok=True)
            
            
            for map_idx in range(2):
                # Create figure with 2 columns (original image + uncertainty map)
                fig, axes = plt.subplots(
                    n_timesteps, 
                    2,
                    figsize=(12, 5 * n_timesteps)
                )
                
                # Handle single-row case
                if n_timesteps == 1:
                    axes = axes.reshape(1, -1)
                
                for row_idx, ts in enumerate(timesteps):
                    # Left column: Original image at full resolution
                    ax_img = axes[row_idx, 0]
                    ax_img.imshow(images[sample_idx])
                    ax_img.set_ylabel(
                        f"t = {ts}",
                        rotation=0,
                        labelpad=40,
                        va="center",
                        fontsize=20
                    )
                    ax_img.set_xticks([])
                    ax_img.set_yticks([])
                    for spine in ax_img.spines.values():
                        spine.set_visible(False)
                    
                    # Right column: Uncertainty map resized to 512x512
                    ax_map = axes[row_idx, 1]
                    uncertainty = uncertainty_maps[ts][last_layer_idx].chunk(2)[map_idx][sample_idx].squeeze(0)

                   
                    
                    if uncertainty.max().item() == uncertainty.min().item():
                        uncertainty *= 0
                    else:
                        pass
                        #uncertainty = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min())
                    
                    # Resize uncertainty map to match original image size (512x512)
                    uncertainty_resized = torch.nn.functional.interpolate(
                        uncertainty.unsqueeze(0).unsqueeze(0),
                        size=(512, 512),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze()
                    
                    ax_map.imshow(uncertainty_resized, cmap='hot')
                    ax_map.set_xticks([])
                    ax_map.set_yticks([])
                    for spine in ax_map.spines.values():
                        spine.set_visible(False)
                
                map_type = 'uncond' if map_idx == 0 else 'cond'
                plt.suptitle(
                    f'Sample {sample_idx+start_idx} - {map_type.upper()} - Last Layer', 
                    fontsize=20, 
                    fontweight='bold',
                    y=0.98
                )
                plt.tight_layout(rect=[0.08, 0.02, 1, 0.96])
                
                enlarged_file_name = f'enlarged_sample_{sample_idx+start_idx}_{map_type}.png'
                plt.savefig(
                    os.path.join(enlarged_vis_dir, enlarged_file_name),
                    dpi=dpi,
                    bbox_inches='tight'
                )
                plt.close()
           



def save_uncertainty_maps(
        uncertainty_maps, 
        sample_idx_copy,
        latents_lst,
        out_dir = None,
        cmap="hot",
        dpi=150,
    
    ):
    
    timesteps = sorted(uncertainty_maps.keys(), reverse=True)
   
    example_ts = timesteps[0]
    last_layer = sorted(uncertainty_maps[example_ts].keys())[-1]

    num_samples = int(uncertainty_maps[example_ts][last_layer].shape[0] / 2)
    #print(len(latents_lst))
    #print(latents_lst[0].shape)
    #exit(1)
    for sample_idx in range(num_samples):
        for map_idx in range(1):
            for row_idx, ts in enumerate(timesteps):
                uncertainty = uncertainty_maps[ts][last_layer].chunk(2)[map_idx][sample_idx].squeeze(0)
                save_dir = f"{out_dir}/{sample_idx+sample_idx_copy}"
                #os.makedirs(save_dir, exist_ok=True)
                ext  = "uncond" if map_idx == 0 else "cond"
                torch.save(uncertainty, f"{save_dir}/{ts}_unmap.pt")
                

                latent = latents_lst[row_idx][sample_idx]
                torch.save(latent, f"{save_dir}/{ts}_latent.pt")

                
                '''
                plt.imshow(uncertainty, cmap='hot')  # cmap can be 'hot', 'viridis', etc.
                plt.colorbar()             # optional: adds a color scale

                # Save as image
                plt.savefig(f"{save_dir}/vis_{ts:04d}_{ext}.jpg", dpi=150, bbox_inches='tight')  # dpi can be adjusted
                plt.close()  # close the figure to free memory'''



           



class HeatmapEvalDataset(Dataset):
    def __init__(self, output_dir):
        self.samples = []
        for entry in os.listdir(output_dir):
            full_path = os.path.join(output_dir, entry)
            if os.path.isdir(full_path):
                image_path = f"{full_path}/output.jpg"
                prompt_path = f"{full_path}/prompt.txt"
                if os.path.exists(image_path) and os.path.exists(prompt_path):
                    with open(prompt_path, "r") as f:
                        prompt = f.readline().strip()
                    self.samples.append({
                        'image_path': image_path,
                        'prompt': prompt,
                        'output_dir': full_path
                    })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_tensor = preprocess_image(sample['image_path'])
        return {
            'image': image_tensor,
            'prompt': sample['prompt'],
            'output_dir': sample['output_dir'],
            'image_path': sample['image_path']
        }

  

def check_results_exists(output_dir, method):

    
    global_file = f"{output_dir}/res.json"
    local_dir = f"{output_dir}/{method}"
    

    if os.path.isfile(global_file):
        with open(global_file) as f:
            data = json.load(f)
            if method in data:
                print(f"Results for {method} were found! (global)")
                return True
    else:
        local_file = f"{local_dir}/res.json"
        if os.path.isfile(local_file):
            with open(local_file) as f:
                data = json.load(f)
                if all(k in data for k in ("FID", "precision", "recall")):
                    print(f"Results for {method} were found! (local)")
                    return True
    
    os.makedirs(local_dir, exist_ok=True)
  

    return False





def visualize_best_worst_per_method(uncertaintity_maps, output_dir, images_path):
    #os.makedirs("grad_cfg_uncertaintity/SD_uncertaintity/output1.jpg", exist_ok=True)
    
    # Load the sample image
    
    for method in uncertaintity_maps:
        method_agg_types = uncertaintity_maps[method]
        agg_types = list(method_agg_types.keys())
        
        # Calculate total rows needed (each agg_type has: 1 image row + 1 uncertainty row + maybe 1 binary row)
        total_rows = 0
        row_info = []
        
        for agg_type in agg_types:
            method_agg_types_dict = method_agg_types[agg_type]
            has_binary = method_agg_types_dict['uncertaintity_maps_bin'][0] is not None if 0 in method_agg_types_dict['uncertaintity_maps_bin'] else False
            has_map = method_agg_types_dict['uncertaintity_maps'][0] is not None if 0 in method_agg_types_dict['uncertaintity_maps'] else False
            if has_binary and has_map:
                num_rows = 3
            elif has_map:
                num_rows =2
            else:
                num_rows = 1

            #num_rows = 3 if has_binary else 2  # image row + uncertainty row + optional binary row
            row_info.append((agg_type, num_rows, has_binary))
            total_rows += num_rows
        
        # Create figure with all rows (8 columns for top-4 and lowest-4)
        fig, axes = plt.subplots(total_rows, 8, figsize=(24, 5 * total_rows))
        if total_rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f'{method}', fontsize=20, fontweight='bold', y=0.995)
        
        current_row = 0
        
        for agg_type, num_rows, has_binary in row_info:
            method_agg_types_dict = method_agg_types[agg_type]
            
            # Get scores and sort to find top-4 and lowest-4
            scores_dict = method_agg_types_dict['uncertaintity_maps_dict']
            sorted_samples = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
            
            top_4 = sorted_samples[:4]
            lowest_4 = sorted_samples[-4:]
            
            # Add agg_type header as text annotation spanning the row
            fig.text(0.5, 1 - ((current_row + 0.3) / total_rows), f'{agg_type}', 
                    ha='center', fontsize=16, fontweight='bold', 
                    transform=fig.transFigure)
            
            # First row: Display sample images (top-4 on left, lowest-4 on right)
            
            for idx, (sample_id, score) in enumerate(top_4):
                axes[current_row, idx].imshow(Image.open(images_path[sample_id]))
                axes[current_row, idx].axis('off')

            for idx, (sample_id, score) in enumerate(lowest_4):
                axes[current_row, idx+4].imshow(Image.open(images_path[sample_id]))
                axes[current_row, idx+4].axis('off')
                
            
            current_row += 1
            
            # Second row: Plot top-4 uncertainty maps
            if has_map:
                for idx, (sample_id, score) in enumerate(top_4):

                    uncertainty_map = method_agg_types_dict['uncertaintity_maps'][sample_id]
                    if isinstance(uncertainty_map, torch.Tensor):
                        uncertainty_map = uncertainty_map.cpu().numpy()
                    
                    axes[current_row, idx].imshow(uncertainty_map, cmap='hot')
                    axes[current_row, idx].set_title(f'Score: {score:.4f}', fontsize=12)
                    axes[current_row, idx].axis('off')
                
                # Plot lowest-4 uncertainty maps
                for idx, (sample_id, score) in enumerate(lowest_4):
                    uncertainty_map = method_agg_types_dict['uncertaintity_maps'][sample_id]
                    if isinstance(uncertainty_map, torch.Tensor):
                        uncertainty_map = uncertainty_map.cpu().numpy()
                    
                    axes[current_row, idx + 4].imshow(uncertainty_map, cmap='hot')
                    axes[current_row, idx + 4].set_title(f'Score: {score:.4f}', fontsize=12)
                    axes[current_row, idx + 4].axis('off')
                
                current_row += 1
            
            # Third row (if binary maps exist): Plot binary maps
            if has_binary:
                for idx, (sample_id, score) in enumerate(top_4):
                    binary_map = method_agg_types_dict['uncertaintity_maps_bin'][sample_id]
                    if isinstance(binary_map, torch.Tensor):
                        binary_map = binary_map.cpu().numpy()
                    
                    axes[current_row, idx].imshow(binary_map, cmap='gray')
                    axes[current_row, idx].axis('off')
                
                for idx, (sample_id, score) in enumerate(lowest_4):
                    binary_map = method_agg_types_dict['uncertaintity_maps_bin'][sample_id]
                    if isinstance(binary_map, torch.Tensor):
                        binary_map = binary_map.cpu().numpy()
                    
                    axes[current_row, idx + 4].imshow(binary_map, cmap='gray')
                    axes[current_row, idx + 4].axis('off')
                
                current_row += 1
        
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        # Save figure
        output_path = os.path.join(output_dir, f'{method}.jpg')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f'Saved: {output_path}')
 
 
   

    #Important - if there are clear losers in terms of aggregation - eliminate them already
    #Next step (important) - implement the global comparison and visualize
    #Next step is the boss - get a real hold of how MAD works, if we're only dealing with one time accelration
    #THE REAL NEXT - be able to incporporate MAD to testing






def vis_metrics_for_methods(x, methods_dict, compare_mode = None, dirs_dict=None,resize_fid=None, images_path = None):
    compare_mode = None
    output_dir = dirs_dict["compare_vis_dir"]
    timesteps_lst = methods_dict["timesteps_basic"][2:-2]
    timesteps_vis = [timesteps_lst[i] for i in np.linspace(0, len(timesteps_lst)-1, 10, dtype=int)]

    d_vis = {}

    if resize_fid == 299:
        resize_fid = None
    
    for method in methods_dict["methods"]:
        final_method = ""
        #d_vis[method] = {}

        if methods_dict["methods"][method]:
            if "ASCED" not in method:
                method_name = "basic"
                agg_types = methods_dict["agg_calculation"] if method == "perTimestep" else methods_dict['global_agg_calculation']
                for agg_calculation in agg_types:
                    semi_final_method = f"{method_name}_{agg_calculation}_{method}"
                    semi_method_to_save = f"{method_name}_{method}"
                   
                    if method == "perTimestep":     
                        for timestep in timesteps_vis:
                            final_method = f"{semi_final_method}_{timestep}"
                            method_to_save = f"{semi_method_to_save}_{timestep}"
                            if method_to_save not in d_vis:
                                d_vis[method_to_save] = {}

                           
                            d_vis[method_to_save][agg_calculation] = generate_map_wrapper(x, final_method, methods_dict, dirs_dict = dirs_dict, compare_mode = None, resize_fid = resize_fid, vis = True)
                    else:
                        final_method = semi_final_method
                        method_to_save = semi_method_to_save
                        if method_to_save not in d_vis:
                            d_vis[method_to_save] = {}
                        d_vis[method_to_save][agg_calculation] = generate_map_wrapper(x, final_method, methods_dict, dirs_dict = dirs_dict, compare_mode = None, resize_fid =resize_fid,vis = True, start_timestep = 12, end_timestep=40)
            else:
                pass

    #print(d_vis['basic_perTimestep_881'].keys())
    visualize_best_worst_per_method(d_vis, output_dir, images_path)
    exit(1)


def eval_metrics_for_methods(x, methods_dict, compare_mode = None, dirs_dict=None,resize_fid=None):

    output_dir = dirs_dict["output_dir_compare"]

   
    if resize_fid == 299:
        resize_fid = None
    
    for method in methods_dict["methods"]:
        final_method = ""

        if methods_dict["methods"][method]:
            if "ASCED" not in method:
                method_name = "basic"
                agg_types = methods_dict["agg_calculation"] if method == "perTimestep" else methods_dict['global_agg_calculation']
                for agg_calculation in agg_types:
                    semi_final_method = f"{method_name}_{agg_calculation}_{method}"
                   
                    if method == "perTimestep":     
                        for timestep in methods_dict["timesteps_basic"]:
                            final_method = f"{semi_final_method}_{timestep}"
                            if check_results_exists(output_dir, final_method) == False:
                                generate_map_wrapper(x, final_method, methods_dict, dirs_dict = dirs_dict, compare_mode = compare_mode, resize_fid = resize_fid)

                    else:
                        final_method = semi_final_method
                        if check_results_exists(output_dir, final_method) == False:
                            generate_map_wrapper(x, final_method, methods_dict, dirs_dict = dirs_dict, compare_mode = compare_mode, resize_fid =resize_fid)
            else:
                method_name = "ASCED"
                agg_types = methods_dict["agg_ASCED_calculation"]
                MAD_start_indices = methods_dict["MAD_start_indices"]
                MAD_end_indices = methods_dict["MAD_end_indices"]
                MAD_values = methods_dict["MAD_values"]


                for agg_calculation in agg_types:
                    for start_idx in MAD_start_indices:
                        for end_idx in MAD_end_indices:
                            for mad_value in MAD_values:
                                final_method = f"{method_name}_{agg_calculation}_{method}_{start_idx}${end_idx}${mad_value}"
                                
                                if check_results_exists(output_dir, final_method) == False:
                                    generate_map_wrapper(x, final_method, methods_dict, dirs_dict = dirs_dict, 
                                                            compare_mode = compare_mode, 
                                                            resize_fid =resize_fid,
                                                            start_timestep = start_idx,
                                                            end_timestep = end_idx,
                                                            mad_value = mad_value,
                                                            asced = True)

                

            
            
            
        #maps = generate_map_single_step(x)






'''
In general - we determine global uncertaintity by:
(1) sum over uncertaintity map
(2) number of pixels above threshold

All possible methods:
(single)
(1) above otsu / median for some single step



to refine on globals:
(2) sum on each map -> take the one with max(sum) V
(3) max on each map -> take the one with max(max) V
(4) sum over time  steps[i,j] -> (2d map) take those above average/otsu -> sum
(5) max over timesteps[i,j] -> (2d map) take those above average/otsu -> sum

(6) above otsu / median for the one with the max value of uncertaintity
(7) above otsu / median for the one with the max(sum) value of uncertaintity

------------------------
(acceleration) - need to try on multiple start-end and multiple MAD values
(1) max difference (for any pixel) between consequetive
(2) some specific step
(3) max difference in terms of sum
(4) 


'''
