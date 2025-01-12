import os
import pickle
import numpy as np
# from PIL import Image
# from collections import Counter
# from pycocotools.coco import COCO
# import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter
from pycocotools.coco import COCO

import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pack_padded_sequence

import Utility as util
import model

image_file_path = 'sample.jpg'
 

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 

def load_image(image_file_path, transform=None):
    img = Image.open(image_file_path).convert('RGB')
    img = img.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        img = transform(img).unsqueeze(0)
    
    return img
 

# Image preprocessing
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.485, 0.456, 0.406), 
                         (0.229, 0.224, 0.225))])


# Load vocabulary wrapper
with open('Data/vocabulary.pkl', 'rb') as f:
    vocabulary = pickle.load(f)


# Build models
encoder_model = model.CNNModel(256).eval()  # eval mode (batchnorm uses moving mean/variance)
decoder_model = model.LSTModel(256, 512, len(vocabulary), 1)
encoder_model = encoder_model.to(device)
decoder_model = decoder_model.to(device)


# Load the trained model parameters
encoder_model.load_state_dict(torch.load('models/encoder-4-3000.ckpt'))
decoder_model.load_state_dict(torch.load('models/decoder-4-3000.ckpt'))


# Prepare an image
img = util.sample_image(image_file_path, transform)
img_tensor = img.to(device)


# Generate an caption from the image
feat = encoder_model(img_tensor)
sampled_indices = decoder_model.sample(feat)
sampled_indices = sampled_indices[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)


# Convert word_ids to words
predicted_caption = []
for token_index in sampled_indices:
    word = vocabulary.i2w[token_index]
    predicted_caption.append(word)
    if word == '<end>':
        break
predicted_sentence = ' '.join(predicted_caption)


# Print out the image and the generated caption
print (predicted_sentence)