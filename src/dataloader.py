import glob
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from PIL import Image
from skimage.color import lab2rgb, rgb2lab
from torch import nn, optim
from torch.utils import data
from torch.utils.data import DataLoader, Dataset, dataloader
from torchvision import transforms
from torchvision.utils import make_grid

size = 128
class ColorizationDataset(Dataset):
    def __init__(self, paths, split = 'train'):
        if split == 'train':
            self.transforms = transforms.Compose([
                transforms.Resize((size, size), Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                ])
        elif split == 'val':
            self.transforms = transforms.Resize((size, size), Image.BICUBIC)
        
        self.split = split
        self.size = size
        self.paths = paths
    
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transforms(img)
        img = np.array(img)
        img_lab = rgb2lab(img).astype("float32")
        img_lab = transforms.ToTensor()(img_lab)
        L = img_lab[[0], ...]/ 50. -1
        ab = img_lab[[1, 2], ...] / 110. 
        
        return {'L': L, 'ab': ab}
    
    def __len__(self):
        return len(self.paths)
    

        
        
        
