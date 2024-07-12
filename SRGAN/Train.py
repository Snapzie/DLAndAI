import sys
import time

import h5py
from PIL import Image
import io

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt

class Gen_res_block(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(64,64,3,padding='same'),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
            nn.Conv2d(64,64,3,padding='same'),
            nn.BatchNorm2d(64),
        )
    
    def forward(self,x_in):
        x = self.block(x_in)
        return x_in + x
    
class Gen_conv_block(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(64,256,3,padding='same'),
            nn.PixelShuffle(2),
            nn.PReLU(64)
        )
    
    def forward(self,x):
        x = self.block(x)
        return x

class Generator(nn.Module):
    def __init__(self,n_residual_blocks=16,n_conv_blocks=2):
        super().__init__()
        self.conv_1 = nn.Conv2d(3,64,9,padding='same')
        self.prelu = nn.PReLU(64)
        self.residual_blocks = nn.Sequential(*[Gen_res_block() for _ in range(n_residual_blocks)])
        self.conv_2 = nn.Conv2d(64,64,3,padding='same')
        self.bn = nn.BatchNorm2d(64)
        self.conv_blocks = nn.Sequential(*[Gen_conv_block() for _ in range(n_conv_blocks)])
        self.conv_3 = nn.Conv2d(64,3,9,padding='same')
    
    def forward(self,x):
        x = self.conv_1(x)
        x_residual = self.prelu(x)
        x = self.residual_blocks(x_residual)
        x = self.conv_2(x)
        x = self.bn(x)
        x = x_residual + x
        x = self.conv_blocks(x)
        x = self.conv_3(x)
        return x
    
class Disc_block(nn.Module):
    def __init__(self,n,s):
        super().__init__()
        in_size = n if s == 2 else n//2
        padding = 'same' if s == 1 else 1
        self.conv = nn.Conv2d(in_size,n,3,stride=s,padding=padding)
        self.bn = nn.BatchNorm2d(n)
    
    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        return F.leaky_relu(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3,64,3,padding='same')
        block_params = [(64,2),(128,1),(128,2),(256,1),(256,2),(512,1),(512,2)]
        self.blocks = nn.Sequential(*[Disc_block(n,s) for n,s in block_params])
        self.flatten = nn.Flatten()
        self.dense_1 = nn.Linear(512*6*6,1024)
        self.dense_2 = nn.Linear(1024,1)
    
    def forward(self,x):
        x = F.leaky_relu(self.conv(x))
        x = self.blocks(x)
        x = self.flatten(x)
        x = F.leaky_relu(self.dense_1(x))
        x = self.dense_2(x)
        return F.sigmoid(x)

class ImageNet10k(Dataset):
    def __init__(self):
        super().__init__()
        self.dataset = []
        pil_to_bytes_h5 = h5py.File('./Data/images.h5','r')
        for key,value in pil_to_bytes_h5.items():
            image_array = np.array(value[()])
            image_buffer = io.BytesIO(image_array)
            image_pil = Image.open(image_buffer)
            self.dataset.append(image_pil)
        pil_to_bytes_h5.close()
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        im = self.dataset[idx]
        im = transforms.ToTensor()(im)
        crop = transforms.RandomCrop((im_size,im_size))(im)
        y = crop
        x = transforms.Resize((im_size//4,im_size//4),transforms.InterpolationMode.BICUBIC)(crop)
        return x,y

im_size = 96
batch_size = 16

print('Loading data...')
t0 = time.time()
dataset = DataLoader(ImageNet10k(),batch_size=batch_size,shuffle=True,pin_memory=True)
t1 = time.time()
print(f'Data loaded in {t1-t0:2.2f} seconds')
count = 0
for x,y in dataset:
    if count > 10:
        break
    print(x.shape,y.shape)
    count += 1

g_model = Generator()
x = torch.rand((1,3,24,24))
print(f'G_in: {x.shape}')
out = g_model(x)
print(f'G_out: {out.shape}')

disc_model = Discriminator()
x = torch.randn((1,3,96,96))
print(f'Disc_in: {x.shape}')
out = disc_model(x)
print(f'Disc_out: {out.shape}')
