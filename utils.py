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




def plot_uncertintiy_maps(
    uncertainty_maps, 
    images,
    prompts = None,
    out_dir = "uncertaintity_maps",
    target_size=128,
    cmap="hot",
    dpi=150):

    os.makedirs(out_dir, exist_ok=True)
    timesteps = sorted(uncertainty_maps.keys(), reverse=True)
    n_timesteps = len(timesteps)
    example_ts = timesteps[0]

    # Number of samples
    n_cols = len(uncertainty_maps[example_ts])#uncertainty_maps[example_ts][0].shape[0] // 2
    n_cols = 1 + n_cols
   
    num_samples = len(images)

    # Image resize
    img_resize = transforms.Resize((target_size, target_size))

    for sample_idx in range(num_samples):
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
                    
                    
            plt.suptitle(f'Sample {sample_idx}', 
                        fontsize=20, fontweight='bold',y=0.98)
            plt.tight_layout(rect=[0.08, 0.05, 1, 0.95])
            vis_file_name = f'sample_{sample_idx}_uncond.png' if map_idx == 0 else f'sample_{sample_idx}_cond.png'
            plt.savefig(os.path.join(out_dir, f'{vis_file_name}'), dpi=150, bbox_inches='tight')
            plt.close()


        # ------------------
        # Create enlarged visualizations for the last layer
        # ------------------
        enlarged_vis_dir = os.path.join(out_dir, 'enlarged_dir')
        os.makedirs(enlarged_vis_dir, exist_ok=True)
        
        last_layer_idx = n_cols - 2  # Last uncertainty map column
        
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
                    uncertainty = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min())
                
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
                f'Sample {sample_idx} - {map_type.upper()} - Last Layer', 
                fontsize=20, 
                fontweight='bold',
                y=0.98
            )
            plt.tight_layout(rect=[0.08, 0.02, 1, 0.96])
            
            enlarged_file_name = f'enlarged_sample_{sample_idx}_{map_type}.png'
            plt.savefig(
                os.path.join(enlarged_vis_dir, enlarged_file_name),
                dpi=dpi,
                bbox_inches='tight'
            )
            plt.close()
           


