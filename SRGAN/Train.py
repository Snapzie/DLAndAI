import torch
import torch.nn.functional as F
import torch.nn as nn

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
