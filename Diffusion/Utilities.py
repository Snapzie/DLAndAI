import os
import torch
import torchvision
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

def noise_images(x,t,noise_schedule):
    sqrt_alpha_hat = torch.sqrt(noise_schedule.alpha_hat[t])[:,None,None,None]
    sqrt_one_minus_alpha_hat = torch.sqrt(1 - noise_schedule.alpha_hat[t])[:,None,None,None]
    eps = torch.randn_like(x)
    return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps, eps

def sample_timestep(n,noise_schedule):
    return torch.randint(low=1,high=noise_schedule.noise_steps,size=(n,))

def sample(model,noise_schedule,n):
    model.eval()
    with torch.no_grad():
        x = torch.randn((n,3,noise_schedule.img_size,noise_schedule.img_size)).to(noise_schedule.device)
        for i in tqdm(reversed(range(1,noise_schedule.noise_steps)),position=0):
            t = (torch.ones(n) * i).long().to(noise_schedule.device)
            pred_noise = model(x,t)
            alpha = noise_schedule.alpha[t][:,None,None,None]
            alpha_hat = noise_schedule.alpha_hat[t][:,None,None,None]
            beta = noise_schedule.beta[t][:,None,None,None]
            if i > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * pred_noise) + torch.sqrt(beta) * noise
    model.train()
    x = (x.clamp(-1,1) + 1) / 2
    x = (x * 255).type(torch.uint8)
    return x

def plot_images(images):
    plt.figure(figsize=(32,32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()],dim=-1),
    ], dim=-2).permute(1,2,0).cpu())
    plt.show()

def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1,2,0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

def get_data(dataset_path,batch_size,image_size):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(80), # im_size + (im_size * 1/4)
        torchvision.transforms.RandomResizedCrop(image_size,scale=(0.8,1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(dataset_path,transform=transforms)
    data_loader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
    return data_loader

def setup_logging(run_name):
    os.makedirs("models",exist_ok=True)
    os.makedirs("results",exist_ok=True)
    os.makedirs(os.path.join("models",run_name), exist_ok=True)
    os.makedirs(os.path.join("results",run_name), exist_ok=True)