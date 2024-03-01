import torch
import torch.nn as nn
from Models import UNet,Noise_schedule
from torch import optim
from Utilities import *
from tqdm import tqdm
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

_epochs = 500
_batch_size = 6
_image_size = 64
_dataset_path = Path("C:/Users/caspe/iCloudDrive/Documents/Projects/DLAndAI/Diffusion/Images")
# _dataset_path = Path("C:/Users/caspe/Desktop/Images")
_lr = 3e-4
_device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", _device)

run_name = 'Diffusion'

setup_logging(run_name)
data_loader = get_data(_dataset_path,_batch_size,_image_size)
model = UNet().to(_device)
optimizer = optim.AdamW(model.parameters(),lr=_lr)
loss_fn = nn.MSELoss()
ns = Noise_schedule(image_size=_image_size,device=_device)
logger = SummaryWriter(os.path.join("runs",run_name))
data_len = len(data_loader)

for epoch in range(_epochs):
    pbar = tqdm(data_loader,desc=f'Processing Epoch {epoch:02d}')
    for i, (images,_) in enumerate(pbar):
        images = images.to(_device)
        t = sample_timestep(images.shape[0],ns).to(_device)
        x_t, noise = noise_images(images,t,ns)
        pred_noise = model(x_t,t)
        loss = loss_fn(noise,pred_noise)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_postfix({f"MSE":f"{loss.item():.4f}"})
        logger.add_scalar("MSE", loss.item(), global_step=epoch*data_len+1)
    
    sampled_images = sample(model,ns,images.shape[0])
    save_images(sampled_images,os.path.join("results",run_name,f"{epoch}.jpg"))
    torch.save(model.state_dict(),os.path.join("models",run_name,f"ckpt.pt"))