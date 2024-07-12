import torch
import torch.nn.functional as F
import torch.nn as nn

class Res_block(nn.Module):
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
    
class Conv_block(nn.Module):
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
        self.residual_blocks = nn.Sequential(*[Res_block() for _ in range(n_residual_blocks)])
        self.conv_2 = nn.Conv2d(64,64,3,padding='same')
        self.bn = nn.BatchNorm2d(64)
        self.conv_blocks = nn.Sequential(*[Conv_block() for _ in range(n_conv_blocks)])
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

g_model = Generator()
x = torch.rand((1,3,24,24))
print(f'G_in: {x.shape}')
out = g_model(x)
print(f'G_out: {out.shape}')

