import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self,image_size,channels,embedding_dim):
        super(Encoder,self).__init__()
        self.conv1 = nn.Conv2d(channels,32,kernel_size=3,stride=2,padding=1)
        self.conv2 = nn.Conv2d(32,64,kernel_size=3,stride=2,padding=1)
        self.conv3 = nn.Conv2d(64,128,kernel_size=3,stride=2,padding=1)

        self.shape_before_flattening = None
        flattened_size = (image_size // 8) * (image_size // 8) * 128
        self.fc = nn.Linear(flattened_size,embedding_dim)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        self.shape_before_flattening = x.shape[1:]

        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x

class Decoder(nn.Module):
    def __init__(self,embedding_dim,shape_before_flattening,channels):
        super(Decoder,self).__init__()

        self.fc = nn.Linear(embedding_dim,np.prod(shape_before_flattening))
        self.reshape_dim = shape_before_flattening

        self.deconv1 = nn.ConvTranspose2d(128,128,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(128,64,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(64,32,kernel_size=3,stride=2,padding=1,output_padding=1)
        self.conv1 = nn.Conv2d(32,channels,kernel_size=3,stride=1,padding=1)

    def forward(self,x):
        x = self.fc(x)
        x = x.view(x.size(0), *self.reshape_dim)

        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))

        x = torch.sigmoid(self.conv1(x))
        return x