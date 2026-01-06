import torch
#from diffusers import StableDiffusionPipeline
from diffusers import DDIMScheduler
from modules.pipeline_stable_diffusion import StableDiffusionPipeline
from modules.pipeline_stable_xl_diffusion import StableDiffusionXLPipeline
from modules.pipeline_pixart_sigma import PixArtSigmaPipeline
from datasets import Dataset
#from diffusers import PixArtSigmaPipeline
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
    "timesteps_basic": ['921', '901', '881', '861', '841', '821', '801', '781', '761', '741', '721', '701', '681', '661', '641', '621', '601', '581', '561', '541', '521', '501', '481', '461', '441', '421', '401', '381', '361', '341', '321', '301', '281', '261', '241', '221', '201', '181', '161', '141', '121', '101', '81', '61', '41' ], #
    "agg_calculation": ["sum", "max",  ], # "aboveAvg", "aboveOtsu"
    'global_agg_calculation': ["pr", "diff", "diffWeighted","prWeighted",    ], # "sumEach$max",  "sumEach$otsu",  "sumOver$sum", #"maxEach$max", "maxEach$otsu", "maxOver$sum"
    "agg_ASCED_calculation": ["sum",  "count"], #"max",

    "global_start_indices": [3, 6, 12, 20],
    "global_end_indices": [-5,-8,-12], #calculated as naegative (-5)


    "MAD_values" : [3,6,10,], #1, 12
    "MAD_start_indices": [10, 12, 14, 20] ,

    "MAD_end_indices": [24, 27, 30,] , # 32, 40
    



    "methods": {

        "perTimestep": False,
        "globalTimestep": True,


        "ASCEDOurs": False,
        
        "ASCEDLatent": False
       
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

            if args.generate_var_uc_scores:
                args.apply_uc = False if args.generate_var_uc_scores else True, 
                args.apply_uc_on_all_timesteps=False if args.generate_var_uc_scores else True, 
                args.return_mid_reps = False if args.generate_var_uc_scores else True

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

    elif args.model == "PixArt":
        scheduler = DDIMScheduler.from_config("PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", subfolder="scheduler",torch_dtype=torch.float16,)
        args.pipe = PixArtSigmaPipeline.from_pretrained(
            "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", 
            torch_dtype=torch.float16,
            scheduler = scheduler,
            use_safetensors=True,
        ).to("cuda")

        

        
        args.batch_size = 2
        


    #if args.generate_var_uc_scores:
    #    args.batch_size = 1

    if args.mode == "compare_methods":
        args.methods_eval = METHODS_EVAL


        if args.compare_vis:
            for method in args.methods_eval["methods"]:
                args.methods_eval["methods"][method] = False
            args.methods_eval["methods"]["globalTimestep"] = True
            args.methods_eval['global_agg_calculation'] = ["prWeighted", "pr"]
            args.methods_eval["global_start_indices"] = [12]
            args.methods_eval["global_end_indices"] = [-5]


        if args.use_global:
            for method in args.methods_eval["methods"]:
                args.methods_eval["methods"][method] = False
            args.methods_eval["methods"]["globalTimestep"] = True
            return


        if args.vis_score_dist:
            args.methods_eval["timesteps_basic"] = ["441","921","881","841", ]
            for method in args.methods_eval["methods"]:
                args.methods_eval["methods"][method] = False
            
            args.methods_eval['global_agg_calculation'] = ["prWeighted"]
            #args.methods_eval["methods"]["globalTimestep"] = True
            args.methods_eval["methods"]["ASCEDLatent"] = True
            args.methods_eval["global_start_indices"] = [3]
            args.methods_eval["global_end_indices"] = [-5]
            
            args.methods_eval["MAD_values"] = [3]
            args.methods_eval["MAD_start_indices"] = [10]
            args.methods_eval["MAD_end_indices"] = [24]



        args.methods_eval["timesteps_basic"] = args.methods_eval["timesteps_basic"][::2]
        if '441' not in args.methods_eval["timesteps_basic"]:
            args.methods_eval["timesteps_basic"].append('441')

        if args.agg_method:

            for method in args.methods_eval["methods"]:
                args.methods_eval["methods"][method] = False
                
            args.methods_eval["methods"]["perTimestep"] = True
            args.methods_eval["agg_calculation"] =  [args.agg_method] 
  

        if args.agg_MAD_method:

            for method in args.methods_eval["methods"]:
                args.methods_eval["methods"][method] = False
                
            args.methods_eval["methods"]["ASCEDLatent"] = True
            args.methods_eval["agg_ASCED_calculation"] =  [args.agg_MAD_method] 


        #print(args.pipe)
        #exit(1)
       
      
