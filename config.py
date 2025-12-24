import torch
#from diffusers import StableDiffusionPipeline
from diffusers import DDIMScheduler
from modules.pipeline_stable_diffusion import StableDiffusionPipeline
from modules.pipeline_stable_xl_diffusion import StableDiffusionXLPipeline
from datasets import Dataset

from torchmetrics.multimodal import CLIPScore
from PIL import Image
from datasets import load_dataset
from modules.unet_2D_conditioned import UNet2DConditionModel
from modules.scheduling_pndm import PNDMScheduler
from torchvision import transforms
import os
import torch.nn.functional as F
import pandas as pd
import json


#need all timesteps for ASCED as well as MAD values and intervals (and of course per timestep as well)
#for finding 


METHODS_EVAL = {
    "timesteps_basic": ['921', '901', '881', '861', '841', '821', '801', '781', '761', '741', '721', '701', '681', '661', '641', '621', '601', '581', '561', '541', '521', '501', '481', '461', '441', '421', '401', '381', '361', '341', '321', '301', '281', '261', '241', '221', '201', '181', '161', '141', '121', '101', '81', '61', '41'],
    "agg_calculation": ["sum", "max", "aboveAvg", "aboveOtsu"], 
    'global_agg_calculation': ["sumEach$max", "maxEach$max", "sumEach$otsu", "maxEach$otsu" "sumOver$sum", "maxOver$sum"],
    "MAD_values" : [],
    
    "methods": {

        "perTimestep": True,
        "globalTimestep": False,


        "ASCED_ours_per_timestep": False,
        "ASCED_ours_per_timestep_jump_two": False,
        "ASCED_latent_per_timestep": False
       
    }
}

def set_config(args, gen_samples = False):
    if gen_samples:
        if args.dataset == "flickr8k":
            args.loaded_dataset = load_dataset("jxie/flickr8k", split=f"validation[:{1000}]", trust_remote_code=True)
        elif args.dataset == "coco":
            coco_dir = "datasets/coco"
            data_file = f'{coco_dir}/annotations/captions_val2014.json'
            data = json.load(open(data_file))

            # merge images and annotations

            images = data['images']
            
            annotations = data['annotations']
            df = pd.DataFrame(images)
            df_annotations = pd.DataFrame(annotations)
            df = df.merge(pd.DataFrame(annotations), how='left', left_on='id', right_on='image_id')

            # keep only the relevant columns
            df = df[['file_name', 'caption']]

            # remove duplicate images
            df = df.drop_duplicates(subset='file_name')
            
            args.loaded_dataset = Dataset.from_pandas(df.reset_index(drop=True))
            args.loaded_dataset = args.loaded_dataset.select(range(30000))

           

            #print(df.head(5))
            #print("Total rows:", len(df))


    if args.model == "1.5v":
        #deterministic(2024)
        unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet",torch_dtype=torch.float16,)
        #scheduler =  PNDMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler",torch_dtype=torch.float16,)
        scheduler = DDIMScheduler.from_config("runwayml/stable-diffusion-v1-5", subfolder="scheduler",torch_dtype=torch.float16,)

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

    if args.mode == "compare_methods":
        args.methods_eval = METHODS_EVAL

        #print(args.pipe)
        #exit(1)
       
      
