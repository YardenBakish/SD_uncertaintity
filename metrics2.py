import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import inception_v3
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from glob import glob


class ImageFolderDataset(Dataset):
    """Simple dataset that loads images from a folder"""
    def __init__(self, folder_path, transform=None, file_indices = None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_files = [f for f in os.listdir(folder_path) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
  
        if len(self.image_files)  == 0:
            self.image_files = [
                file for file in glob(os.path.join(folder_path, f"**/*/*.jpg"), recursive=True)
            ]

        if file_indices:
            pref = "/".join(self.image_files[0].split("/")[:-2])
            suff = self.image_files[0].split("/")[-1]
            self.image_files = [f"{pref}/{elem}/{suff}" for elem in file_indices]
        
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


def get_inception_model(device):
    """Load pretrained Inception v3 model for feature extraction"""
    model = inception_v3(pretrained=True, transform_input=False)
    model.fc = torch.nn.Identity()  # Remove final classification layer
    model = model.to(device)
    model.eval()
    return model


def extract_features(image_folder, model, device, batch_size=32, extract_features = None, file_indices =None):
    """Extract Inception features from all images in a folder"""
    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = ImageFolderDataset(image_folder, transform=transform, file_indices=file_indices)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=4, pin_memory=True)
    
    features_list = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Extracting features"):
            batch = batch.to(device)
            feats = model(batch)
            features_list.append(feats.cpu())
    
    features = torch.cat(features_list, dim=0)
    return features


def compute_pairwise_distances(row_features, col_features, batch_size=100):
    """Compute pairwise distances between two feature sets"""
    num_rows = row_features.shape[0]
    distances = []
    
    for i in range(0, num_rows, batch_size):
        batch = row_features[i:i+batch_size]
        dist_batch = torch.cdist(batch.unsqueeze(0), col_features.unsqueeze(0))[0]
        distances.append(dist_batch)
    
    return torch.cat(distances, dim=0)


def compute_precision_recall(real_features, gen_features, nhood_size=3):
    """
    Compute precision and recall metrics
    
    Args:
        real_features: Features from real images [N_real, feature_dim]
        gen_features: Features from generated images [N_gen, feature_dim]
        nhood_size: Neighborhood size (k in k-NN)
    
    Returns:
        precision: Float value for precision
        recall: Float value for recall
    """
    print(f"\nComputing precision and recall with neighborhood size k={nhood_size}")
    print(f"Real images: {real_features.shape[0]}, Generated images: {gen_features.shape[0]}")
    
    # Compute precision (how many generated images fall in real manifold)
    print("\nComputing precision...")
    real_distances = compute_pairwise_distances(real_features, real_features)
    # Get k-th nearest neighbor distance for each real sample
    kth_real = torch.kthvalue(real_distances, nhood_size + 1, dim=1).values
    
    # Check which generated samples fall within real manifold
    gen_to_real_dist = compute_pairwise_distances(gen_features, real_features)
    precision = (gen_to_real_dist <= kth_real.unsqueeze(0)).any(dim=1).float().mean().item()
    
    # Compute recall (how many real images are covered by generated manifold)
    print("Computing recall...")
    gen_distances = compute_pairwise_distances(gen_features, gen_features)
    kth_gen = torch.kthvalue(gen_distances, nhood_size + 1, dim=1).values
    
    # Check which real samples are covered by generated manifold
    real_to_gen_dist = compute_pairwise_distances(real_features, gen_features)
    recall = (real_to_gen_dist <= kth_gen.unsqueeze(0)).any(dim=1).float().mean().item()
    
    return precision, recall


def calculate_metrics(real_folder, gen_folder, nhood_size=3, batch_size=32, device=None, file_indices = None):
    """
    Main function to calculate precision and recall
    
    Args:
        real_folder: Path to folder containing real images
        gen_folder: Path to folder containing generated images
        nhood_size: Neighborhood size for k-NN (default: 3)
        batch_size: Batch size for feature extraction (default: 32)
        device: Device to run on (default: auto-detect)
    
    Returns:
        dict with 'precision' and 'recall' keys
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Load Inception model
    print("\nLoading Inception v3 model...")
    model = get_inception_model(device)
    
    # Extract features
    print(f"\nExtracting features from real images in: {real_folder}")

    real_features = extract_features(real_folder, model, device, batch_size) #real_folder
    
    print(f"\nExtracting features from generated images in: {gen_folder}")
    gen_features = extract_features(gen_folder, model, device, batch_size, file_indices=file_indices)
    
    # Compute metrics
    precision, recall = compute_precision_recall(real_features, gen_features, nhood_size)
    
    return {
        'precision': precision,
        'recall': recall
    }

'''
if __name__ == "__main__":
    # Example usage
    real_folder = "datasets/coco/val2014"
    gen_folder = "datasets/coco/val2014"
    
    results = calculate_metrics(
        real_folder=real_folder,
        gen_folder=gen_folder,
        nhood_size=3,
        batch_size=32
    )
    
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {2 * results['precision'] * results['recall'] / (results['precision'] + results['recall']):.4f}")'''