import glob
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from PIL import Image
from skimage.color import lab2rgb, rgb2lab
from torch import nn, optim
from torch.optim import optimizer
from torch.utils import data
from torch.utils.data import DataLoader, Dataset, dataloader
from torchvision import transforms
from torchvision.utils import make_grid

from dataloader import ColorizationDataset
from model import *
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path = "C:/Users/chauh/Documents/Github/tutorial_notebooks/ImageCaptioning/input/train2014/"

paths = glob.glob(path+"/*.jpg")

np.random.seed(9123)

paths_subset = np.random.choice(paths, 12_000, replace = False)
rand_idxs = np.random.permutation(12_000)
train_idxs = rand_idxs[:10000]
val_idxs = rand_idxs[10000:]
train_paths = paths_subset[train_idxs]
val_paths = paths_subset[val_idxs]

_, axes = plt.subplots(4, 4, figsize=(10, 10))
for ax, img_path in zip(axes.flatten(), train_paths):
    ax.imshow(Image.open(img_path))
    ax.axis("off")


size = 128

class MainModel(nn.Module):
    def __init__(self, net_G=None, lr_G=2e-4, lr_D=2e-4, beta1=0.5, beta2=0.999, lambda_L1=100.):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )
        self.lambda_L1 = lambda_L1
        
        if net_G is None:
            self.net_G = init_model(Unet(input_c=1, output_c=2, n_down=8, num_filters=64), self.device)
        else:
            self.net_G = net_G.to(self.device)
        self.net_D = init_model(PatchDiscriminator(input_c=3, n_down=3, num_filters=64), self.device)
        self.GANcriterion = GANLoss(gan_mode='vanilla').to(self.device)
        self.L1criterion = nn.L1Loss()
        self.opt_G = optim.Adam(self.net_G.parameters(), lr=lr_G, betas=(beta1, beta2))
        self.opt_D = optim.Adam(self.net_D.parameters(), lr=lr_D, betas=(beta1, beta2))
    
    def set_requires_grad(self, model, requires_grad = True):
        for p in model.parameters():
            p.requires_grad = requires_grad
            
    def setup_input(self, data):
        self.L = data['L'].to(device)
        self.ab = data['ab'].to(self.device)
    
    def forward(self):
        self.fake_color = self.net_G(self.L)
    
    def backward_D(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image.detach())
        self.loss_D_fake = self.GANcriterion(fake_preds, False)
        real_image = torch.cat([self.L, self.ab], dim=1)
        real_preds = self.net_D(real_image)
        self.loss_D_real = self.GANcriterion(real_preds, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()
    
    def backward_G(self):
        fake_image = torch.cat([self.L, self.fake_color], dim=1)
        fake_preds = self.net_D(fake_image)
        self.loss_G_GAN = self.GANcriterion(fake_preds, True)
        self.loss_G_L1 = self.L1criterion(self.fake_color, self.ab) * self.lambda_L1
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()
    
    def optimize(self):
        self.forward()
        self.net_D.train()
        self.set_requires_grad(self.net_D, True)
        self.opt_D.zero_grad()
        self.backward_D()
        self.opt_D.step()
        
        self.net_G.train()
        self.set_requires_grad(self.net_D, False)
        self.opt_G.zero_grad()
        self.backward_G()
        self.opt_G.step()

def pretrain_generator(net_G, train_dl, optimizer, criterion, epochs):
    for e in range(epochs):
        loss_meter = AverageMeter()
        for data in tqdm.tqdm(train_dl):
            L, ab = data['L'].to(device), data['ab'].to(device)
            preds = net_G(L)
            loss = criterion(preds, ab)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_meter.update(loss.item(), L.size(0))
            
        print(f'Epoch {e +1}/{epochs} ')
        print(f'L1 Loss: {loss_meter.avg:.5f}')

def train_model(model, train_dl, epochs, display_every=200):
    data = next(iter(train_dl)) # getting a batch for visualizing the model output after fixed intrvals
    for e in range(epochs):
        loss_meter_dict = create_loss_meters() # function returning a dictionary of objects to 
        i = 0                                  # log the losses of the complete network
        for data in tqdm.tqdm(train_dl):
            model.setup_input(data) 
            model.optimize()
            update_losses(model, loss_meter_dict, count=data['L'].size(0)) # function updating the log objects
            i += 1
            if i % display_every == 0:
                print(f"\nEpoch {e+1}/{epochs}")
                print(f"Iteration {i}/{len(train_dl)}")
                log_results(loss_meter_dict) # function to print out the losses
                visualize(model, data, save=False) # function displaying the model's outputs      
        if e == (epochs-1):
            torch.save(model.state_dict(), "final-trained-model.pt")

def valid_model(model, valid_dl):
    model.load_state_dict(torch.load('final-trained-model.pt', map_location='cuda'))
        
    model.eval()
    data = next(iter(valid_dl))
    loss_meter_dict = create_loss_meters()
    with torch.no_grad():
        for data in tqdm.tqdm(valid_dl):
            model.setup_input(data)
            # update_losses(model, loss_meter_dict, count=data['L'].size(0)) # function updating the log objects
            # print(f"\nEpoch {epochs}")
            # log_results(loss_meter_dict) # function to print out the losses
            visualize(model, data, save=False) # function displaying the model's outputs      
        
    
        
if __name__ == '__main__':
    train_dl = make_dataloaders(paths=train_paths, split = 'train')
    val_dl = make_dataloaders(paths=val_paths, split = 'val')

    # net_G = build_res_unet(n_input=1, n_output=2, size=256)
    # optimizer = optim.Adam(net_G.parameters(), lr=1e-4)
    # criterion = nn.L1Loss()
    # pretrain_generator(net_G, train_dl, optimizer, criterion, 20)
    # torch.save(net_G.state_dict(), "res18-unet.pt")
    
    net_G = build_res_unet(n_input=1, n_output=2, size=256)
    net_G.load_state_dict(torch.load("res18-unet.pt", map_location=device))
    model = MainModel(net_G=net_G)
    # train_model(model, train_dl, 20)
    valid_model(model, val_dl)
