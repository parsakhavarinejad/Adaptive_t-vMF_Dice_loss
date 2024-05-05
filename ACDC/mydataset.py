import numpy as np
import torch
import torch.utils.data as data
from torchvision import datasets, transforms
import os
from PIL import Image, ImageOps
import random

class CustomImageMaskDataset(data.Dataset):
    def __init__(self, dataframe, image_transform=None):
        self.data = dataframe
        self.image_transform = image_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.iloc[idx]['image_path']
        mask_path = self.data.iloc[idx]['mask_path']

        image = Image.open(image_path).convert('RGB') 
        mask = Image.open(mask_path).convert('RGB')
        
        seed = np.random.randint(2147483647)
        random.seed(seed) 
        torch.manual_seed(seed) 
        if self.image_transform:
            image = self.image_transform(image)
        
        random.seed(seed) 
        torch.manual_seed(seed)
        if self.image_transform:
            mask = self.image_transform(mask)

        return image, mask


