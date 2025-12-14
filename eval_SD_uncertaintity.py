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
from utils import *
import matplotlib.pyplot as plt

import argparse

NUM_SAMPLES_TO_GENERATE = 20

preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--mode', type=str, default = 'demo', choices = ['generate_uncertaintity_samples', 'demo'])
    parser.add_argument('--generation_method', type=str, default = 'basic', choices = ['basic'])

    args = parser.parse_args()
    args.output_dir = f"uncertaintity_maps/{args.generation_method}"
    args.batch_size = 16
    os.makedirs(args.output_dir, exist_ok=True)
    return args





def demo(args):

    unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet",torch_dtype=torch.float16,)
    scheduler =  PNDMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler",torch_dtype=torch.float16,)
    # Load model (weights download automatically)
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        unet=unet,  
        scheduler = scheduler,
    ).to("cuda")


    dataset = load_dataset("jxie/flickr8k", split="validation[:5]", trust_remote_code=True)  # take 5 examples for demo
    prompts = [item["caption_0"] for item in dataset]
 
    output = pipe(prompts, apply_uc = True)

    images = output[0].images
    uncertainty_maps = output[1]

    for idx in range(len(images)):
        print(prompts[idx])
        images[idx].save(f"output{idx}.jpg", quality=95)


    plot_uncertintiy_maps(
        uncertainty_maps, 
        images,
        prompts,
        out_dir = "uncertaintity_maps_demo",
        target_size=128,
        cmap="hot",
        dpi=150)




def generate_uncertaintity_samples(args):

    unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet",torch_dtype=torch.float16,)
    scheduler =  PNDMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler",torch_dtype=torch.float16,)
    # Load model (weights download automatically)
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        unet=unet,  
        scheduler = scheduler,
    ).to("cuda")


    dataset = load_dataset("jxie/flickr8k", split=f"validation[:{NUM_SAMPLES_TO_GENERATE}]", trust_remote_code=True) 

    sample_idx = 0
    for batch_start in range(0, len(dataset), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(dataset))
        batch_items = dataset[batch_start:batch_end]
        
        # Select shortest caption for each item in batch
        prompts = []
        for i in range(len(batch_items['caption_0'])):
            # Get all captions for this item
            captions = [batch_items[f'caption_{j}'][i] for j in range(5) if f'caption_{j}' in batch_items]
            # Select shortest
            shortest_caption = min(captions, key=len)
            prompts.append(shortest_caption)
        
       

        # Generate images
        output = pipe(prompts, apply_uc=True, apply_uc_on_all_timesteps = True)
        images = output[0].images
        uncertainty_maps = output[1]
        
        # Save each image with its prompt
        for idx in range(len(images)):
            # Create subdirectory for this sample
            sample_dir = os.path.join(args.output_dir, str(sample_idx))
            os.makedirs(sample_dir, exist_ok=True)
            
            # Save prompt to txt file
            with open(os.path.join(sample_dir, "prompt.txt"), "w") as f:
                f.write(prompts[idx])
            
            # Save image
            images[idx].save(os.path.join(sample_dir, "output.jpg"), quality=95)
            
            print(f"Sample {sample_idx}: {prompts[idx]}")
            sample_idx += 1
        exit(1)
    




if __name__ == "__main__":
    args          = parse_args()
    if args.mode == "demo":
        demo(args)
    elif args.mode == "generate_uncertaintity_samples":
        generate_uncertaintity_samples(args)




'''
# Convert PIL images to tensors for CLIPScore
preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])



image_tensors = torch.stack([preprocess(img) for img in images]).to("cuda")

# Evaluate with CLIPScore
clip_metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to("cuda")
scores = clip_metric(image_tensors, prompts)

print("CLIP scores:", scores)'''
































'''
# Access the actual model components for visualization
unet = pipe.unet  # This is where you can add hooks/visualizations
vae = pipe.vae
text_encoder = pipe.text_encoder

#print(pipe)
#exit(1)

# Generate images
prompts = ["a photo of a dog kicking a ninja", "a photo of a cat", "a landscape painting"]
images = pipe(prompts).images
images[0].save("output.jpg", quality=95)
images[1].save("output1.jpg", quality=95)
images[2].save("output2.jpg", quality=95)

# Evaluate with CLIP
image_tensors = torch.stack([preprocess(img) for img in images]).to("cuda")

clip_metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to("cuda")
for idx, img in enumerate(image_tensors):
    #print(image_tensors.shape)
    #print(img.shape)
    #exit(1)
    scores = clip_metric(img, prompts[idx])
    print(f"CLIP scores: {scores}")'''