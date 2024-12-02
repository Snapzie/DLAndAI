import os
from glob import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def visualize_and_save_batch(epoch,lr_imgs, gen_imgs, hr_imgs, output_dir="output"):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for idx, (lr_img, gen_img, hr_img) in enumerate(zip(lr_imgs, gen_imgs, hr_imgs)):
        # Move tensors to CPU and convert to numpy
        lr_img = lr_img * 0.5 + 0.5
        lr_img = lr_img.cpu().numpy().transpose(1, 2, 0)
        gen_img = gen_img.cpu().detach().numpy().transpose(1, 2, 0)
        hr_img = hr_img * 0.5 + 0.5
        hr_img = hr_img.cpu().numpy().transpose(1, 2, 0)

        # Normalize generated image from [-1, 1] to [0, 1]
        gen_img = (gen_img - gen_img.min()) / (gen_img.max() - gen_img.min())

        # Plot and save the images
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.title("Low-Resolution")
        plt.imshow(lr_img)
        plt.subplot(1, 3, 2)
        plt.title("Generated High-Resolution")
        plt.imshow(gen_img)
        plt.subplot(1, 3, 3)
        plt.title("Ground Truth High-Resolution")
        plt.imshow(hr_img)

        # Save the figure
        plt.savefig(os.path.join(output_dir, f"result_{epoch}-{idx}.png"))
        plt.close()

class SuperResolutionDataset(Dataset):
    def __init__(self, hr_dir, lr_dir):
        self.hr_images = sorted(glob(f"{hr_dir}/*.png"))
        self.lr_images = sorted(glob(f"{lr_dir}/*.png"))
        self.hr_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.lr_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, idx):
        hr_image = Image.open(self.hr_images[idx])
        lr_image = Image.open(self.lr_images[idx])
        return self.lr_transform(lr_image), self.hr_transform(hr_image)

# Dataset and DataLoader
train_dataset = SuperResolutionDataset("./Images_HR", "./Images_LR")
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, num_residual_blocks=16):
        super(Generator, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )
        self.residual_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_residual_blocks)])
        self.upsampling = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )
        self.final = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        x = self.initial(x)
        res = x
        x = self.residual_blocks(x)
        x = self.upsampling(x + res)
        return self.final(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        def block(in_channels, out_channels, stride):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2)
            )

        self.model = nn.Sequential(
            block(3, 64, 1),
            block(64, 64, 2),
            block(64, 128, 1),
            block(128, 128, 2),
            block(128, 256, 1),
            block(256, 256, 2),
            block(256, 512, 1),
            block(512, 512, 2),
            # nn.Flatten(),
            # nn.Linear(512 * (4 * 4), 1024),
            # nn.LeakyReLU(0.2),
            # nn.Linear(1024, 1)
        )

        # Dynamically infer the flattened size
        self.flatten_size = None
        self._initialize_flatten_size()

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_size, 1024),  # Replace 8192 with dynamically computed size
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1)
        )

    def _initialize_flatten_size(self):
        # Use a dummy input to compute the flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 256, 256)  # Input size (3, 256, 256)
            x = dummy_input
            for layer in self.model:
                x = layer(x)
            self.flatten_size = x.numel()

    def forward(self, x):
        return self.model(x)

from torchvision.models import vgg19

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = vgg19(pretrained=True).features[:36]
        self.model = vgg.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, gen_img, target_img):
        return torch.mean((self.model(gen_img) - self.model(target_img)) ** 2)

import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
device = 'cuda'
epochs = 101

# Models
generator = Generator().to(device)
# torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
discriminator = Discriminator().to(device)

generator.compile()
discriminator.compile()

# Losses
pixel_loss = nn.MSELoss()
adversarial_loss = nn.BCEWithLogitsLoss()
content_loss = PerceptualLoss().to(device)

# Optimizers
gen_optimizer = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.9,0.999))
disc_optimizer = optim.Adam(discriminator.parameters(), lr=2e-4,betas=(0.9,0.999))


# Load checkpoint
checkpoint = torch.load("checkpoint.pth", map_location=device)

# Restore model states
generator.load_state_dict(checkpoint["generator_state_dict"])
discriminator.load_state_dict(checkpoint["discriminator_state_dict"])

# Restore optimizer states
gen_optimizer.load_state_dict(checkpoint["gen_optimizer_state_dict"])
disc_optimizer.load_state_dict(checkpoint["disc_optimizer_state_dict"])

# Restore scaler state if using mixed precision
scaler.load_state_dict(checkpoint["scaler_state_dict"])

# Restore epoch
epoch = checkpoint["epoch"] + 1


# Training Loop
for epoch in range(epochs):
    for lr_imgs, hr_imgs in train_loader:
        lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)

        # Train Discriminator
        disc_optimizer.zero_grad()
        with autocast():  # Mixed precision enabled
            real_preds = discriminator(hr_imgs)
            fake_imgs = generator(lr_imgs)
            fake_preds = discriminator(fake_imgs.detach())
            real_loss = adversarial_loss(real_preds, torch.ones_like(real_preds) * 0.9) # Label smoothing
            fake_loss = adversarial_loss(fake_preds, torch.zeros_like(fake_preds))
            disc_loss = real_loss + fake_loss
        
        # Scale gradients and perform the backward pass
        scaler.scale(disc_loss).backward()
        scaler.step(disc_optimizer)
        scaler.update()

        # Train Generator
        gen_optimizer.zero_grad()
        with autocast():  # Mixed precision enabled
            fake_preds = discriminator(fake_imgs)
            adv_loss = adversarial_loss(fake_preds, torch.ones_like(fake_preds))
            pix_loss = pixel_loss(fake_imgs, hr_imgs)
            perc_loss = content_loss(fake_imgs, hr_imgs)
            gen_loss = adv_loss * 0.001 + 0.15 * perc_loss + 0.1 * pix_loss
        
        # Scale gradients and perform the backward pass
        scaler.scale(gen_loss).backward()
        scaler.step(gen_optimizer)
        scaler.update()

    if epoch % 20 == 0:
        # Save weights and optimizer states
        torch.save({
            "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "gen_optimizer_state_dict": gen_optimizer.state_dict(),
            "disc_optimizer_state_dict": disc_optimizer.state_dict(),
            "epoch": epoch, # Save the current epoch for tracking
            "scaler_state_dict": scaler.state_dict() # For mixed precision
        }, "checkpoint.pth")

        # Example
        lr_imgs, hr_imgs = next(iter(train_loader))  # Get a batch of images
        gen_imgs = generator(lr_imgs.to(device))     # Generate high-resolution images

        # Save all images in the batch
        visualize_and_save_batch(epoch,lr_imgs, gen_imgs, hr_imgs)

    print(f"Epoch {epoch + 1}, Gen Loss: {gen_loss.item()}, Disc Loss: {disc_loss.item()}")




