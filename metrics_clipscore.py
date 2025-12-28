import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import inception_v3
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from glob import glob
from functools import partial

from torchmetrics.multimodal import CLIPScore


class ImageFolderDataset(Dataset):
    """Simple dataset that loads images from a folder"""
    def __init__(self, folder_path,  file_indices = None):
        self.folder_path = folder_path
        self.image_files = [f for f in os.listdir(folder_path) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
  
        if len(self.image_files)  == 0:
            self.image_files = [
                file for file in glob(os.path.join(folder_path, f"**/*/*.jpg"), recursive=True)
            ]
        #print(self.image_files[0])
        
        if file_indices:
            pref = "/".join(self.image_files[0].split("/")[:-2])
            suff = self.image_files[0].split("/")[-1]
            self.image_files = [f"{pref}/{elem}/{suff}" for elem in file_indices]
        
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
        
        image = Image.open(img_path).convert('RGB')
        image = np.array(image).astype("uint8")
       
        
        return self.prompts[idx], image



def numpy_collate(batch):
    prompts, images = zip(*batch)
    images = np.stack(images, axis=0)   # batch as NumPy
    return list(prompts), images


def calculate_clipscore_metric(image_folder, batch_size=32, device=None, file_indices = None):
   
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #from torchmetrics.functional.multimodal import clip_score
    clip_model = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(device)

    clip_model.eval()

    #clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

    dataset = ImageFolderDataset(image_folder, file_indices=file_indices)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=0, pin_memory=True, collate_fn=numpy_collate)
    
    
    scores = []
    with torch.no_grad():
        for prompts, images in tqdm(dataloader, desc=f"Extracting features"):
          
            clip_score = clip_model(torch.from_numpy(images).permute(0, 3, 1, 2).to(device), prompts).detach()
            scores.append(clip_score.cpu())
            
    
    sum_score = sum(scores)
    len_score = len(scores)
    return (sum_score / len_score).item()
'''

if __name__ == "__main__":
    # Example usage
    image_folder = "uncertaintity_maps/SDXL"

    calculate_clipscore_metric(image_folder, batch_size=256,)
    

    print(results)'''