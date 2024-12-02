# import the necessary packages
import matplotlib
import numpy as np
import torch
import torchvision
from models import config
matplotlib.use("agg")
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from tqdm import tqdm

def extract_random_images(data_loader, num_images):
    # initialize empty lists to store all images and labels
    all_images = []
    all_labels = []
    # iterate through the data loader to get images and labels
    for images, labels in data_loader:
        # append the current batch of images and labels to the respective lists
        all_images.append(images)
        all_labels.append(labels)
        # stop the iteration if the total number of images exceeds 1000
        if len(all_images) * data_loader.batch_size > 1000:
            break
    # concatenate all the images and labels tensors along the 0th dimension
    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    # generate random indices for selecting a subset of images and labels
    random_indices = np.random.choice(len(all_images), num_images, replace=False)
    # use the random indices to extract the corresponding images and labels
    random_images = all_images[random_indices]
    random_labels = all_labels[random_indices]
    # return the randomly selected images and labels to the calling function
    return random_images, random_labels

def display_images(images, labels, num_images_per_row, title, filename=None, show=True):
    # calculate the number of rows needed to display all the images
    num_rows = len(images) // num_images_per_row
    # create a grid of images using torchvision's make_grid function
    grid = torchvision.utils.make_grid(
        images.cpu(), nrow=num_images_per_row, padding=2, normalize=True
    )
    # convert the grid to a NumPy array and transpose it to
    # the correct dimensions
    grid_np = grid.numpy().transpose((1, 2, 0))
    # create a new figure with the appropriate size
    plt.figure(figsize=(num_images_per_row * 2, num_rows * 2))
    # show the grid of images
    plt.imshow(grid_np)
    # remove the axis ticks
    plt.axis("off")
    # set the title of the plot
    plt.title(title, fontsize=16)

        # add labels for each image in the grid
    for i in range(len(images)):
        # calculate the row and column of the current image in the grid
        row = i // num_images_per_row
        col = i % num_images_per_row
        # get the name of the label for the current image
        label_name = config.CLASS_LABELS[labels[i].item()]
        # add the label name as text to the plot
        plt.text(
            col * (images.shape[3] + 2) + images.shape[3] // 2,
            (row + 1) * (images.shape[2] + 2) - 5,
            label_name,
            fontsize=12,
            ha="center",
            va="center",
            color="white",
            bbox=dict(facecolor="black", alpha=0.5, lw=0),
        )
    # if show is True, display the plot
    if show:
        plt.show()
    else:
        # otherwise, save the plot to a file and close the figure
        plt.savefig(filename, bbox_inches="tight")
        plt.close()

def display_random_images(
    data_loader,
    encoder=None,
    decoder=None,
    file_recon=None,
    file_real=None,
    title_recon=None,
    title_real=None,
    display_real=True,
    num_images=32,
    num_images_per_row=8,
):
    # extract a random subset of images and labels from the data loader
    random_images, random_labels = extract_random_images(data_loader, num_images)
        # if an encoder and decoder are provided,
    # use them to generate reconstructions
    if encoder is not None and decoder is not None:
        # set the encoder and decoder to evaluation mode
        encoder.eval()
        decoder.eval()
        # move the random images to the appropriate device
        random_images = random_images.to(config.DEVICE)
        # generate embeddings for the random images using the encoder
        random_embeddings = encoder(random_images)
        # generate reconstructions for the random images using the decoder
        random_reconstructions = decoder(random_embeddings)
        # display the reconstructed images
        display_images(
            random_reconstructions.cpu(),
            random_labels,
            num_images_per_row,
            title_recon,
            file_recon,
            show=False,
        )
        # if specified, also display the original images
        if display_real:
            display_images(
                random_images.cpu(),
                random_labels,
                num_images_per_row,
                title_real,
                file_real,
                show=False,
            )
    # if no encoder and decoder are provided, simply display the original images
    else:
        display_images(
            random_images, random_labels, num_images_per_row, title="Real Images"
        )


def validate(encoder, decoder, test_loader, criterion):
    # set the encoder and decoder to evaluation mode
    encoder.eval()
    decoder.eval()
    # initialize the running loss to 0.0
    running_loss = 0.0
    # disable gradient calculation during validation
    with torch.no_grad():
        # iterate through the test loader
        for batch_idx, (data, _) in tqdm(
            enumerate(test_loader), total=len(test_loader)
        ):
            # move the data to the appropriate device CPU/GPU
            data = data.to(config.DEVICE)
            # encode the data using the encoder
            encoded = encoder(data)
            # decode the encoded data using the decoder
            decoded = decoder(encoded)
            # calculate the loss between the decoded and original data
            loss = criterion(decoded, data)
            # add the loss to the running loss
            running_loss += loss.item()
    # calculate the average loss over all batches
    # and return to the calling function
    return running_loss / len(test_loader)

def get_test_embeddings(test_loader, encoder):
    # switch the model to evaluation mode
    encoder.eval()
    # initialize empty lists to store the embeddings and labels
    points = []
    label_idcs = []
    # iterate through the test loader
    for i, data in enumerate(test_loader):
        # move the images and labels to the appropriate device
        img, label = [d.to(config.DEVICE) for d in data]
        # encode the test images using the encoder
        proj = encoder(img)
        # convert the embeddings and labels to NumPy arrays
        # and append them to the respective lists
        points.extend(proj.detach().cpu().numpy())
        label_idcs.extend(label.detach().cpu().numpy())
        # free up memory by deleting the images and labels
        del img, label
    # convert the embeddings and labels to NumPy arrays
    points = np.array(points)
    label_idcs = np.array(label_idcs)
    # return the embeddings and labels to the calling function
    return points, label_idcs

def plot_latent_space(test_loader, encoder, show=False):
    # get the embeddings and labels for the test images
    points, label_idcs = get_test_embeddings(test_loader, encoder)
    # create a new figure and axis for the plot
    fig, ax = plt.subplots(figsize=(10, 10) if not show else (8, 8))
    # create a scatter plot of the embeddings, colored by the labels
    scatter = ax.scatter(
        x=points[:, 0],
        y=points[:, 1],
        s=2.0,
        c=label_idcs,
        cmap="tab10",
        alpha=0.9,
        zorder=2,
    )
    # remove the top and right spines from the plot
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # add a colorbar to the plot
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.ax.set_ylabel("Labels", rotation=270, labelpad=20)
    # if show is True, display the plot
    if show:
        # add a grid to the plot
        ax.grid(True, color="lightgray", alpha=1.0, zorder=0)
        plt.show()
    # otherwise, save the plot to a file and close the figure
    else:
        plt.savefig(config.LATENT_SPACE_PLOT, bbox_inches="tight")
        plt.close()

def get_random_test_images_embeddings(test_loader, encoder, imgs_visualize=5000):
    # get all the images and labels from the test loader
    all_images, all_labels = [], []
    for batch in test_loader:
        images_batch, labels_batch = batch
        all_images.append(images_batch)
        all_labels.append(labels_batch)
    # concatenate all the images and labels into a single tensor
    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    # randomly select a subset of the images and labels to visualize
    index = np.random.choice(range(len(all_images)), imgs_visualize)
    images = all_images[index]
    labels = all_labels[index]
    # get the embeddings for all the test images
    points, _ = get_test_embeddings(test_loader, encoder)
    # select the embeddings corresponding to the randomly selected images
    embeddings = points[index]
    # return the randomly selected images, their labels, and their embeddings
    return images, labels, embeddings

def plot_image_grid_on_embeddings(
    test_loader, encoder, decoder, grid_size=15, figsize=12, show=True
):
    # get a random subset of test images
    # and their corresponding embeddings and labels
    _, labels, embeddings = get_random_test_images_embeddings(test_loader, encoder)
    # create a single figure for the plot
    fig, ax = plt.subplots(figsize=(figsize, figsize))
    # define a custom color map with discrete colors for each unique label
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    cmap = cm.get_cmap("rainbow", num_classes)
    bounds = np.linspace(0, num_classes, num_classes + 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    # Plot the scatter plot of the embeddings colored by label
    scatter = ax.scatter(
        embeddings[:, 0],
        embeddings[:, 1],
        cmap=cmap,
        c=labels,
        norm=norm,
        alpha=0.8,
        s=300,
    )
    # Create the colorbar with discrete ticks corresponding to unique labels
    cb = plt.colorbar(scatter, ticks=range(num_classes), spacing="proportional", ax=ax)
    cb.set_ticklabels(unique_labels)

    # Create the grid of images to overlay on the scatter plot
    x = np.linspace(embeddings[:, 0].min(), embeddings[:, 0].max(), grid_size)
    y = np.linspace(embeddings[:, 1].max(), embeddings[:, 1].min(), grid_size)
    xv, yv = np.meshgrid(x, y)
    grid = np.column_stack((xv.ravel(), yv.ravel()))
    # convert the numpy array to a PyTorch tensor
    # and get reconstructions from the decoder
    grid_tensor = torch.tensor(grid, dtype=torch.float32)
    reconstructions = decoder(grid_tensor.to(config.DEVICE))
    # overlay the images on the scatter plot
    for i, (grid_point, img) in enumerate(zip(grid, reconstructions)):
        img = img.squeeze().detach().cpu().numpy()
        imagebox = OffsetImage(img, cmap="Greys", zoom=0.5)
        ab = AnnotationBbox(
            imagebox, grid_point, frameon=False, pad=0.0, box_alignment=(0.5, 0.5)
        )
        ax.add_artist(ab)
  
    plt.show()