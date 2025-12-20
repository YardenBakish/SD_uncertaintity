import torch
#from diffusers import StableDiffusionPipeline
from diffusers import DDIMScheduler
from modules.pipeline_stable_diffusion import StableDiffusionPipeline
from modules.pipeline_stable_xl_diffusion import StableDiffusionXLPipeline

from torchmetrics.multimodal import CLIPScore
from PIL import Image
from datasets import load_dataset
from modules.unet_2D_conditioned import UNet2DConditionModel
from modules.scheduling_pndm import PNDMScheduler
from torchvision import transforms
import os
import torch.nn.functional as F

def set_config(args):
    if args.model == "1.5v":
        #deterministic(2024)
        unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet",torch_dtype=torch.float16,)
        #scheduler =  PNDMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler",torch_dtype=torch.float16,)
        #scheduler = DDIMScheduler.from_config("runwayml/stable-diffusion-v1-5", subfolder="scheduler",torch_dtype=torch.float16,)

        # Load model (weights download automatically)
        args.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            unet=unet,  
        # scheduler = scheduler,
        ).to("cuda")
        args.batch_size = 8
    elif args.model == "SDXL":
        unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="unet",torch_dtype=torch.float16,)
        scheduler = DDIMScheduler.from_config("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler",torch_dtype=torch.float16,)
        args.pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", scheduler=scheduler, torch_dtype=torch.float16, use_safetensors=True, variant="fp16", unet=unet).to("cuda")
        args.batch_size = 2

        #print(args.pipe)
        #exit(1)
       
      
