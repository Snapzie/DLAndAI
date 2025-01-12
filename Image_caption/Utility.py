import os
import nltk
from PIL import Image
from collections import Counter
from pycocotools.coco import COCO
 
import torch
import torch.utils.data as data

####### Vocabulary #######
# nltk.download('punkt')
# nltk.download('punkt_tab')

class Vocab(object):
    def __init__(self):
        self.w2i = {}
        self.i2w = {}
        self.index = 0
    
    def __call__(self,token):
        if not token in self.w2i:
            return self.w2i['<unk>']
        return self.w2i[token]
    
    def __len__(self):
        return len(self.w2i)
    
    def add_token(self,token):
        if not token in self.w2i:
            self.w2i[token] = self.index
            self.i2w[self.index] = token
            self.index += 1

def build_vocabulary(json,threshold):
    coco = COCO(json)
    counter = Counter()
    ids = coco.anns.keys()
    for i, id in enumerate(ids):
        caption = str(coco.anns[id]['caption'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

        if (i+1) % 1000 == 0:
            print("[{}/{}] Tokenized the captions.".format(i+1,len(ids)))

    tokens = [token for token,cnt in counter.items() if cnt >= threshold]

    vocab = Vocab()
    vocab.add_token('<pad>')
    vocab.add_token('<start>')
    vocab.add_token('<end>')
    vocab.add_token('<unk>')

    for i, token in enumerate(tokens):
        vocab.add_token(token)
    return vocab


####### Image preprocessing #######
def reshape_image(image,shape):
    return image.resize(shape, Image.LANCZOS)

def reshape_images(image_path,output_path,shape):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    images = os.listdir(image_path)
    num_im = len(images)
    for i, im in enumerate(images):
        with open(os.path.join(image_path,im), 'r+b') as f:
            with Image.open(f) as image:
                image = reshape_image(image,shape)
                image.save(os.path.join(output_path,im),image.format)
        if (i+1) % 100 == 0:
            print(f'[{i+1}/{num_im}] Resized the images and saved into {output_path}')


####### Dataloader #######
class CustomCocoDataset(data.Dataset):
    def __init__(self,data_path,coco_json_path,vocabulary,transform=None):
        self.root = data_path
        self.coco_data = COCO(coco_json_path)
        self.indices = list(self.coco_data.anns.keys())
        self.vocabulary = vocabulary
        self.transform = transform

    def __getitem__(self,idx):
        caption = self.coco_data.anns[self.indices[idx]]['caption']
        image_id = self.coco_data.anns[self.indices[idx]]['image_id']
        image_path = self.coco_data.loadImgs(image_id)[0]['file_name']

        image = Image.open(os.path.join(self.root,image_path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        
        word_tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(self.vocabulary('<start>'))
        caption.extend([self.vocabulary(token) for token in word_tokens])
        caption.append(self.vocabulary('<end>'))
        ground_truth = torch.Tensor(caption)
        return image,ground_truth
    
    def __len__(self):
        return len(self.indices)
    
def collate_function(data_batch):
    data_batch.sort(key=lambda d: len(d[1]),reverse=True)
    imgs,caps = zip(*data_batch)

    imgs = torch.stack(imgs,0)

    cap_lens = [len(cap) for cap in caps]
    tgts = torch.zeros(len(caps), max(cap_lens)).long()
    for i, cap in enumerate(caps):
        end = cap_lens[i]
        tgts[i,:end] = cap[:end]
    return imgs, tgts, cap_lens

def get_loader(data_path,coco_json_path,vocabulary,transform,batch_size,shuffle):
    coco_dataset = CustomCocoDataset(data_path=data_path,coco_json_path=coco_json_path,vocabulary=vocabulary,transform=transform)
    custom_data_loader = torch.utils.data.DataLoader(dataset=coco_dataset,batch_size=batch_size,shuffle=shuffle,collate_fn=collate_function)
    return custom_data_loader

####### Sampling #######
def sample_image(image_file_path, transform=None):
    img = Image.open(image_file_path).convert('RGB')
    img = img.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        img = transform(img).unsqueeze(0)
    
    return img