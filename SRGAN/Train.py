import sys
import time
from tqdm import tqdm

import h5py
from PIL import Image
import io

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
import matplotlib.pyplot as plt
import random

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
    
    def rescale(self,im):
        min_val = torch.abs(torch.min(torch.min(im,2)[0],2)[0])
        min_val = min_val.unsqueeze(2).unsqueeze(3)
        im = im + min_val

        max_val = torch.max(torch.max(im,2)[0],2)[0]
        max_val = max_val.unsqueeze(2).unsqueeze(3)
        im = im/max_val
        return im

    def sample(self,epoch,dataset,n=10):
        x,y = next(iter(dataset))
        y = y.to(device)
        x = x.to(device)
        # g = F.sigmoid(self.forward(x))
        g = g_model(x)
        # g = g + torch.abs(torch.min(g)) # Min val now 0
        # g = g / torch.max(g) # Max val now 1
        g = self.rescale(g)

        for i in range(n):
            fig = plt.figure(figsize=(32,32))
            ax = fig.add_subplot()
            out = g[i,:,:,:]
            out = out.permute(1,2,0).detach().cpu().numpy()
            ax.imshow(out)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.savefig(f'./Samples/img_{epoch}_{i}_gen')
            plt.close()

            fig = plt.figure(figsize=(32,32))
            ax = fig.add_subplot()
            hr = y[i,:,:,:]
            hr = hr.permute(1,2,0).detach().cpu().numpy()
            ax.imshow(hr)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.savefig(f'./Samples/img_{epoch}_{i}_org')
            plt.close()
    
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
    def __init__(self,path):
        super().__init__()
        self.dataset = []
        pil_to_bytes_h5 = h5py.File(path,'r')
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
        y = im
        x = F.interpolate(im.unsqueeze(0),scale_factor=1/4,mode='bicubic').squeeze(0)
        # x = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])(x)
        # y = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])(y)
        x = torch.clamp(x,0,1)
        y = torch.clamp(y,0,1)

        x_h,x_w = x.shape[1],x.shape[2]
        rh = random.randrange(0,x_h-patch_size+1)
        rw = random.randrange(0,x_w-patch_size+1)
        rh_scaled = rh * scale_factor
        rw_scaled = rw * scale_factor

        x = x[:,rh:rh+patch_size,rw:rw+patch_size]
        y = y[:,rh_scaled:rh_scaled+(patch_size*scale_factor),rw_scaled:rw_scaled+(patch_size*scale_factor)]
        return x,y

def vgg_loss(hr,sr):
    vgg(hr)
    vgg_hr = vgg_out['relu5_4']
    vgg(sr)
    vgg_sr = vgg_out['relu5_4']
    return mse(vgg_hr,vgg_sr) * 0.006

def adveserial_training():
    iter = 0
    iterations_per_step = len(train_dset.dataset)
    while iter < iterations:
        g_model.train()
        d_model.train()

        for step,(x,y) in enumerate(train_dset,start=1):
            x = x.to(device)
            y = y.to(device)
            labels_real = torch.ones((x.shape[0],1)).to(device)
            labels_fake = torch.zeros((x.shape[0],1)).to(device)
            
            # Discriminator
            g_optimizer.zero_grad()
            d_optimizer.zero_grad()

            g_images = g_model(x)
            d_fake = d_model(g_images)
            d_real = d_model(y)

            d_loss = bce(d_real,labels_real) + bce(d_fake,labels_fake)
            d_loss.backward()
            d_optimizer.step()

            # Generator
            g_optimizer.zero_grad()
            d_optimizer.zero_grad()

            g_images = g_model(x)
            d_fake = d_model(g_images)

            l_vgg = vgg_loss(y,g_images)
            l_adv = bce(d_fake,labels_real)
            g_loss = l_vgg + l_adv * 1e-3
            g_loss.backward()
            g_optimizer.step()

            print(f'Iteration {iter} {step:3d}/{steps} | G_loss: {g_loss.item():.4f} | D_loss: {d_loss.item():.6f}')
        
        iter += iterations_per_step
        
        if iter % (5 * iterations_per_step) == 0:
            g_model.eval()
            with torch.no_grad():
                g_model.sample(iter,val_dset)

def resnet_training():
    iter = 0
    iterations_per_step = len(train_dset.dataset)
    while iter < iterations:
        g_model.train()
        loss_accum = 0
        pbar = tqdm(train_dset,desc=f'Iteration {int(iter/1e3)}e3',ncols=100)
        for step,(x,y) in enumerate(pbar,start=1):
            x = x.to(device)
            y = y.to(device)
            g_optimizer.zero_grad()

            gen_images = g_model(x)
            # gen_images = gen_images + torch.abs(torch.min(gen_images)) # Min val now 0
            # gen_images = gen_images / torch.max(gen_images) # Max val now 1
            # gen_images = (gen_images*2)-1 # value range now [-1;1]
            gen_images = g_model.rescale(gen_images)
            gen_images = (gen_images*2)-1
            y = (y*2)-1

            assert -1 <= torch.min(gen_images) <= 1, 'gen_images out of range'
            assert -1 <= torch.min(y) <= 1, 'y out of range'

            loss = mse(gen_images,y)
            loss_accum += loss.detach().item()
            loss.backward()
            g_optimizer.step()

            pbar.set_postfix_str(f'Loss: {(loss_accum / step):.6f}')

        if iter % (20 * iterations_per_step) == 0:
            g_model.eval()
            with torch.no_grad():
                val_loss = 0
                pbar = tqdm(val_dset,desc=f'Iteration {int(iter/1e3)}e3',ncols=100)
                for step,(x,y) in enumerate(pbar):
                    x = x.to(device)
                    y = y.to(device)

                    gen_images = g_model(x)
                    # gen_images = gen_images + torch.abs(torch.min(gen_images)) # Min val now 0
                    # gen_images = gen_images / torch.max(gen_images) # Max val now 1
                    # gen_images = (gen_images*2)-1 # value range now [-1;1]
                    gen_images = g_model.rescale(gen_images)
                    gen_images = (gen_images*2)-1
                    y = (y*2)-1

                    val_loss += mse(gen_images,y)
                    pbar.set_postfix_str(f'Val Loss: {(val_loss / step).item():.6f}')
                g_model.sample(iter,val_dset)
            torch.save(g_model.state_dict(), f'./Checkpoints/iter_{iter}.pt')

        iter += iterations_per_step
        
           

im_size = 96
scale_factor = 4
patch_size = 24
batch_size = 128
iterations = 17e6
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using: {device}')

print('Loading data...')
t0 = time.time()
train_dset = DataLoader(ImageNet10k('./Data/training.h5'),batch_size=batch_size,shuffle=True,pin_memory=True)
val_dset = DataLoader(ImageNet10k('./Data/validation.h5'),batch_size=batch_size)
steps = len(train_dset)
t1 = time.time()

g_model = Generator().to(device)
d_model = Discriminator().to(device)

vgg_out = {}
def activation(name):
    def hook(module, input, output):
        vgg_out[name] = output
    return hook

vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).to(device)
for param in vgg.parameters():
    param.requires_grad = False
vgg.features[35].register_forward_hook(activation('relu5_4'))

mse = nn.MSELoss()
bce = nn.BCELoss()
g_optimizer = torch.optim.AdamW(g_model.parameters(),lr=1e-4)
d_optimizer = torch.optim.AdamW(d_model.parameters(),lr=1e-4)

torch.set_float32_matmul_precision('high')

# model.load_state_dict(torch.load(PATH))

# g_model.compile()
# d_model.compile()
# vgg.compile()

resnet_training()
# adveserial_training()




    
        



