import os
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)

LR = 0.001
PATIENCE = 2
IMAGE_SIZE = 32
CHANNELS = 1
BATCH_SIZE = 64
EMBEDDING_DIM = 2
EPOCHS = 10

output_dir = 'output'
os.makedirs(output_dir,exist_ok=True)

training_progress_dir = os.path.join(output_dir,'training_progress')
os.makedirs(training_progress_dir,exist_ok=True)

model_weights_dir = os.path.join(output_dir,'model_weights')
os.makedirs(model_weights_dir,exist_ok=True)

MODEL_WEIGHTS_PATH = os.path.join(model_weights_dir, "best_autoencoder.pt")
FILE_RECON_BEFORE_TRAINING = os.path.join(output_dir, "reconstruct_before_train.png")
FILE_REAL_BEFORE_TRAINING = os.path.join(
    output_dir, "real_test_images_before_train.png"
)
# define reconstruction & real after training images path
FILE_RECON_AFTER_TRAINING = os.path.join(output_dir, "reconstruct_after_train.png")
FILE_REAL_AFTER_TRAINING = os.path.join(output_dir, "real_test_images_after_train.png")
# define latent space and image grid embeddings plot path
LATENT_SPACE_PLOT = os.path.join(output_dir, "embedding_visualize.png")
IMAGE_GRID_EMBEDDINGS_PLOT = os.path.join(output_dir, "image_grid_on_embeddings.png")
# define class labels dictionary
CLASS_LABELS = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}