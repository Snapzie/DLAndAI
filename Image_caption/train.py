import os
import pickle
import numpy as np
# from PIL import Image
# from collections import Counter
# from pycocotools.coco import COCO
# import matplotlib.pyplot as plt
 
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pack_padded_sequence

import Utility as util
import model

# Create vocabulary if vocabulary file does not already exist
if not os.path.exists('Data/vocabulary.pkl'):
    vocab = util.build_vocabulary(json='Data/annotations/captions_train2014.json',threshold=4)
    vocab_path = './Data/vocabulary.pkl'
    with open(vocab_path,'wb') as f:
        pickle.dump(vocab,f)
    print('Total vocabulary size {}'.format(len(vocab)))
    print('Saved the vocabulary wrapper to "{}"'.format(vocab_path))

# Resizing training images if resized images does not already exist
if not os.path.exists('Data/resized_images'):
    image_path = './Data/train2014/'
    output_path = './Data/resized_images/'
    image_shape = [256, 256]
    util.reshape_images(image_path, output_path, image_shape)


# Training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

if not os.path.exists('models/'):
    os.makedirs('models')

transform = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.255))
])

with open('Data/vocabulary.pkl', 'rb') as f:
    vocabulary = pickle.load(f)
custom_data_loader = util.get_loader('Data/resized_images', 'Data/annotations/captions_train2014.json',vocabulary,transform,128,shuffle=True)

encoder_model = model.CNNModel(256).to(device)
decoder_model = model.LSTModel(256,512,len(vocabulary),1).to(device)

loss_criterion = nn.CrossEntropyLoss()
parameters = list(decoder_model.parameters()) + list(encoder_model.linear_layer.parameters()) + list(encoder_model.batch_norm.parameters())
optimizer = torch.optim.Adam(parameters,lr=0.001)

total_num_steps = len(custom_data_loader)
for epoch in range(5):
    for i,(imgs,caps,lens) in enumerate(custom_data_loader):
        encoder_model.train()
        decoder_model.train()

        imgs = imgs.to(device)
        caps = caps.to(device)
        tgts = pack_padded_sequence(caps,lens,batch_first=True)[0]

        feats = encoder_model(imgs)
        outputs = decoder_model(feats,caps,lens)
        loss = loss_criterion(outputs,tgts)
        decoder_model.zero_grad()
        encoder_model.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f'Epoch [{epoch}/{5}], Step [{i}/{total_num_steps}], Loss: {loss.item():.4f}, Perplexity: {np.exp(loss.item()):5.4f}')
        
        if i % 500 == 0:
            encoder_model.eval()
            decoder_model.eval()
            sample_transform = transforms.Compose([
                transforms.ToTensor(), 
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            sample_img = util.sample_image('sample.jpg', sample_transform)
            sample_img_tensor = sample_img.to(device)

            sample_features = encoder_model(sample_img_tensor)
            sampled_indices = decoder_model.sample(sample_features)
            sampled_indices = sampled_indices[0].cpu().numpy()

            predicted_caption = []
            for token_index in sampled_indices:
                word = vocabulary.i2w[token_index]
                predicted_caption.append(word)
                if word == '<end>':
                    break
            predicted_sentence = ' '.join(predicted_caption)
            print(predicted_sentence)

        if (i+1) % 1000 == 0:
            torch.save(decoder_model.state_dict(), os.path.join(
                'models/', 'decoder-{}-{}.ckpt'.format(epoch+1, i+1)))
            torch.save(encoder_model.state_dict(), os.path.join(
                'models/', 'encoder-{}-{}.ckpt'.format(epoch+1, i+1)))