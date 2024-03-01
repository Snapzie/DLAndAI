import torch
import torch.nn as nn
import torch.nn.functional as F

# Inspiration from https://github.com/dome272/Diffusion-Models-pytorch/blob/main/modules.py

class Noise_schedule:
    def __init__(self,noise_steps=1000,beta_start=1e-4,beta_end=0.02,image_size=64,device='cuda'):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = image_size
        self.device = device

        self.beta = torch.linspace(self.beta_start,beta_end,self.noise_steps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha,dim=0)

class DoubleConv(nn.Module):
    def __init__(self,in_c,out_c,mid_channels=None,residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_c
        self.doubleConv = nn.Sequential(
            nn.Conv2d(in_c,mid_channels,3,padding=1,bias=False),
            nn.GroupNorm(1,mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels,out_c,3,padding=1,bias=False),
            nn.GroupNorm(1,out_c)
        )
    
    def forward(self,x):
        if self.residual:
            return F.gelu(x + self.doubleConv(x))
        else:
            return self.doubleConv(x)

class SelfAttention(nn.Module):
    def __init__(self,c,size):
        super().__init__()
        self.c = c
        self.size = size
        self.mha = nn.MultiheadAttention(c,4,batch_first=True)
        self.norm = nn.LayerNorm([c])
        self.feedforward = nn.Sequential(
            nn.LayerNorm([c]),
            nn.Linear(c,c),
            nn.GELU(),
            nn.Linear(c,c)
        )
    
    def forward(self,x):
        x = x.view(-1,self.c,self.size * self.size).swapaxes(1,2)
        x_norm = self.norm(x)
        attn,_ = self.mha(x_norm,x_norm,x_norm)
        attn = attn + x
        attn = self.feedforward(attn) + attn
        return attn.swapaxes(2,1).view(-1,self.c,self.size,self.size)

class Down(nn.Module):
    def __init__(self,in_c,out_c,emb_dim=256):
        super().__init__()
        self.max_pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_c,in_c,residual=True),
            DoubleConv(in_c,out_c)
        )
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim,out_c) # (1,256) --> (1,out_c) ?
        )
    
    def forward(self,x,t):
        x = self.max_pool_conv(x)
        emb = self.emb_layer(t)[:,:,None,None].repeat(1,1,x.shape[-2],x.shape[-1]) # (1,c) --> (1,c,,) --> (1,c,out_c,out_c) ?
        return x + emb # (1,c,out_c,out_c)

class Up(nn.Module):
    def __init__(self,in_c,out_c,emb_dim=256):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_c,in_c,residual=True),
            DoubleConv(in_c,out_c,in_c // 2)
        )
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim,out_c) # (1,256) --> (1,3) ?
        )

    def forward(self,x,skip,t):
        x = self.up(x) # (1,c,x,x) --> (1,c,x*2,x*2) ?
        x = torch.cat([skip,x],dim=1) # (1,c*2,x*2,x*2) ?
        x = self.conv(x)
        emb = self.emb_layer(t)[:,:,None,None].repeat(1,1,x.shape[-2],x.shape[-1]) # (1,c) --> (1,c,,) --> (1,c,out_c,out_c) ?
        return x + emb

class UNet(nn.Module):
    def __init__(self,c_in=3,c_out=3,time_dim=256,device='cuda'):
        super().__init__()
        self.device = device
        self.time_dim = time_dim

        self.first_layer = DoubleConv(c_in,64)
        self.down1 = Down(64,128)
        self.SA1 = SelfAttention(128,32)
        self.down2 = Down(128,256)
        self.SA2 = SelfAttention(256,16)
        self.down3 = Down(256,256)
        self.SA3 = SelfAttention(256,8)

        self.bottle1 = DoubleConv(256,512)
        self.bottle2 = DoubleConv(512,512)
        self.bottle3 = DoubleConv(512,256)

        self.up1 = Up(512,128)
        self.SA4 = SelfAttention(128,16)
        self.up2 = Up(256,64)
        self.SA5 = SelfAttention(64,32)
        self.up3 = Up(128,64)
        self.SA6 = SelfAttention(64,64)

        self.out = nn.Conv2d(64,c_out,kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq) # (1,time_dim // 2)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq) # (1,time_dim // 2)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)          # (1,time_dim)
        return pos_enc
    
    def forward(self,x,t):
        t = t.unsqueeze(-1).type(torch.float)  # (1,)
        t = self.pos_encoding(t,self.time_dim) # (1,time_dim)

        x1 = self.first_layer(x)
        x2 = self.down1(x1,t)
        x2 = self.SA1(x2)
        x3 = self.down2(x2,t)
        x3 = self.SA2(x3)
        x4 = self.down3(x3,t)
        x4 = self.SA3(x4)

        x4 = self.bottle1(x4)
        x4 = self.bottle2(x4)
        x4 = self.bottle3(x4)

        x5 = self.up1(x4,x3,t)
        x5 = self.SA4(x5)
        x5 = self.up2(x5,x2,t)
        x5 = self.SA5(x5)
        x5 = self.up3(x5,x1,t)
        x5 = self.SA6(x5)

        return self.out(x5)



# net = UNet(device='cpu')
# print(sum([p.numel() for p in net.parameters()]))
# x = torch.randn(3, 3, 64, 64)
# t = x.new_tensor([500] * x.shape[0]).long()
# print(net(x, t).shape)