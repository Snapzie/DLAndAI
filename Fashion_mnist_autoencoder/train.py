import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
from models import config, network, utils

transform = transforms.Compose([
    transforms.Pad(padding=2),
    transforms.ToTensor(),
    transforms.Grayscale(1)
])

trainset = datasets.ImageFolder(root="fashion_mnist/train",transform=transform)
train_loader = torch.utils.data.DataLoader(trainset,batch_size=config.BATCH_SIZE,shuffle=True)

testset = datasets.ImageFolder(root="fashion_mnist/test",transform=transform)
test_loader = torch.utils.data.DataLoader(trainset,batch_size=config.BATCH_SIZE,shuffle=True)

encoder = network.Encoder(
    channels=config.CHANNELS,
    image_size=config.IMAGE_SIZE,
    embedding_dim=config.EMBEDDING_DIM
).to(config.DEVICE)

_ = encoder(torch.rand((config.BATCH_SIZE,config.CHANNELS,config.IMAGE_SIZE,config.IMAGE_SIZE)).to(config.DEVICE))
shape_before_flattening = encoder.shape_before_flattening

decoder = network.Decoder(
    embedding_dim=config.EMBEDDING_DIM,
    shape_before_flattening=shape_before_flattening,
    channels=config.CHANNELS
).to(config.DEVICE)

criterion = nn.BCELoss()
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=config.LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",factor=0.1,patience=config.PATIENCE,verbose=True)

utils.display_random_images(
    test_loader,
    encoder,
    decoder,
    title_recon="Reconstructed Before Traning",
    title_real="Real Test Images",
    file_recon=config.FILE_RECON_BEFORE_TRAINING,
    file_real=config.FILE_REAL_BEFORE_TRAINING
)

best_val_loss = float("inf")

for epoch in range(config.EPOCHS):
    print(f"Epoch: {epoch + 1}/{config.EPOCHS}")
    encoder.train()
    decoder.train()

    running_loss = 0.0
    for batch_idx,(data,_) in tqdm(enumerate(train_loader),total=len(train_loader)):
        data = data.to(config.DEVICE)
        optimizer.zero_grad()

        encoded = encoder(data)
        decoded = decoder(encoded)

        loss = criterion(decoded,data)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    val_loss = utils.validate(encoder,decoder,test_loader,criterion)

    print(f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(
            {
                "encoder":encoder.state_dict(),
                "decoder":decoder.state_dict()
            }, config.MODEL_WEIGHTS_PATH
        )
    
    scheduler.step(val_loss)

    utils.display_random_images(
        data_loader=test_loader,
        encoder=encoder,
        decoder=decoder,
        file_recon=os.path.join(config.training_progress_dir,f"epoch{epoch+1}_test_recon.png"),
        display_real=False
    )
print("Training done")