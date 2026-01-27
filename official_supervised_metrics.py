# import
from transformers import AutoProcessor, AutoModel
from PIL import Image
import torch

from glob import glob
from torch.utils.data import Dataset, DataLoader
import ImageReward as RM
from tqdm import tqdm
import numpy as np
import hpsv2
import os
import torch
import torch.nn as nn
from os.path import expanduser  # pylint: disable=import-outside-toplevel
from urllib.request import urlretrieve  # pylint: disable=import-outside-toplevel

import open_clip

import pandas as pd
import json


class ImageFolderDataset(Dataset):
    """Simple dataset that loads images from a folder"""
    def __init__(self, folder_path,  file_indices = None):


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
        df = df['file_name'].to_list()
       
        

        self.folder_path = folder_path
        self.image_files = [f for f in os.listdir(folder_path) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
  
        if len(self.image_files)  == 0:
            self.image_files = [
                file for file in glob(os.path.join(folder_path, f"**/*/*.jpg"), recursive=True)
            ]
        #print(self.image_files[0])


        self.real_files = [f"{coco_dir}/val2014/{elem}" for elem in df][:len(self.image_files)] 
        
        if file_indices:

            self.real_files = np.array(self.real_files)
            self.real_files = self.real_files[file_indices]
            self.real_files  = list(self.real_files )
            pref = "/".join(self.image_files[0].split("/")[:-2])
            suff = self.image_files[0].split("/")[-1]
            self.image_files = [f"{pref}/{elem}/{suff}" for elem in file_indices]
        
     
        #print(len(self.image_files))
        #print(len(self.real_files))
        #exit(1)


        prompt_files = [f"{'/'.join(image_file.split('/')[:-1])}/prompt.txt" for image_file in self.image_files]

        self.prompts = []
        for path in prompt_files:
            with open(path, "r") as f:
                self.prompts.append(f.readline().strip()) 
        

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):


        if "/" not in self.image_files[idx]:
            img_path = os.path.join(self.folder_path, self.image_files[idx])
        else:
            img_path = self.image_files[idx]

        #return self.prompts[idx], img_path
        
        image = img_path #Image.open(img_path).convert('RGB')
        image_ref = self.real_files[idx]#Image.open(self.real_files[idx]).convert('RGB')
        
        return self.prompts[idx], image, image_ref





def pickscore(image_folder, file_indices = None):

    dataset = ImageFolderDataset(image_folder, file_indices=file_indices)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, prefetch_factor=2,
                           num_workers=2, pin_memory=True, )
    
    # load model
    device = "cuda"
    processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
    model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"

    processor = AutoProcessor.from_pretrained(processor_name_or_path)
    model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(device)


    def calc_probs(prompt, images):
        
        # preprocess
        image_inputs = processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)
        
        text_inputs = processor(
            text=prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)


        with torch.no_grad():
            # embed
            image_embs = model.get_image_features(**image_inputs)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
        
            text_embs = model.get_text_features(**text_inputs)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
        
            # score
            scores = model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
            
            # get probabilities if you have multiple images to choose from
            probs = torch.softmax(scores, dim=-1)
        
        return probs.cpu().tolist()
        
    
    scores = []
   
    with torch.no_grad():
        for prompts, pred_image, ref_image in tqdm(dataloader, desc=f"Extracting features"):
           

            pred_image = pred_image[0]
            ref_image = ref_image[0]
            prompts = prompts[0]

           
            pil_images = [Image.open(pred_image), Image.open(ref_image)]
            prompt = prompts

            pred_prob =  calc_probs(prompt, pil_images)
           
            scores.append(pred_prob[0])
    

    summed_scores = sum(scores)
    len_scores = len(scores)
    res = summed_scores / len_scores
   
    return res



def use_hpsv2(image_folder, file_indices = None):
    dataset = ImageFolderDataset(image_folder, file_indices=file_indices)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, prefetch_factor=2,
                           num_workers=2, pin_memory=True, )
    
    scores = []
   
    for prompts, pred_image, ref_image in tqdm(dataloader, desc=f"Extracting features"):
       
        
        pred_image = pred_image[0]
        ref_image = ref_image[0]
        prompts = prompts[0]

        result = hpsv2.score(Image.open(pred_image), prompts, hps_version="v2.1") 
        #print(result)
        result = float(result[0].item())
        scores.append(result)
        #print(result)

    summed_scores = sum(scores)
    len_scores = len(scores)
    return summed_scores / len_scores





def use_RM(image_folder, file_indices = None):
    dataset = ImageFolderDataset(image_folder, file_indices=file_indices)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, prefetch_factor=2,
                           num_workers=2, pin_memory=True, )
    
    scores = []
    
    model = RM.load("ImageReward-v1.0")
    for prompts, pred_image, ref_image in tqdm(dataloader, desc=f"Extracting features"):

        pred_image = pred_image[0]
        ref_image = ref_image[0]
        prompts = prompts[0]
        result = model.score(prompts, [Image.open(pred_image), Image.open(ref_image)])
        result = np.array(result)
        result = (result - result.min()) / (result.max() - result.min())
        result = min(result)
        scores.append(result)

        print(result)

    summed_scores = sum(scores)
    len_scores = len(scores)
    return summed_scores / len_scores





def get_aesthetic_model(clip_model="vit_l_14"):
    """load the aethetic model"""
    home = expanduser("~")
    cache_folder = home + "/.cache/emb_reader"
    path_to_model = cache_folder + "/sa_0_4_"+clip_model+"_linear.pth"
    if not os.path.exists(path_to_model):
        os.makedirs(cache_folder, exist_ok=True)
        url_model = (
            "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_"+clip_model+"_linear.pth?raw=true"
        )
        urlretrieve(url_model, path_to_model)
    if clip_model == "vit_l_14":
        m = nn.Linear(768, 1)
    elif clip_model == "vit_b_32":
        m = nn.Linear(512, 1)
    else:
        raise ValueError()
    s = torch.load(path_to_model)
    m.load_state_dict(s)
    m.eval()
    return m


def use_AES(image_folder, file_indices = None):
    dataset = ImageFolderDataset(image_folder, file_indices=file_indices)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, prefetch_factor=2,
                           num_workers=2, pin_memory=True, )
    
    scores = []
    
    amodel= get_aesthetic_model(clip_model="vit_l_14")
    amodel.eval()

    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')

    for prompts, pred_image, ref_image in tqdm(dataloader, desc=f"Extracting features"):
        pred_image = Image.open(pred_image[0])
        ref_image = ref_image[0]
        prompts = prompts[0]
        image = preprocess(pred_image).unsqueeze(0)
        with torch.no_grad():
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            prediction = float(amodel(image_features)[0].item())
            print(prediction)
            scores.append(prediction)

    summed_scores = sum(scores)
    len_scores = len(scores)
    return summed_scores / len_scores



def run_sup_metrics(gen_folders, file_indices = None):
    pick_score = pickscore(gen_folders, file_indices = file_indices)
    hpsv2_score = 2
    ##RM_score = use_RM(gen_folders, file_indices = filtered)
    aes_score = use_AES(gen_folders, file_indices = file_indices)

    return pick_score, hpsv2_score, aes_score





