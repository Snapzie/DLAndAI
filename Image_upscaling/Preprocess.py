import cv2
import os
from glob import glob

def resize_images(image_dir, output_dir, target_size):
    os.makedirs(output_dir, exist_ok=True)
    images = glob(f"{image_dir}/*.png")  # Assumes PNG format; change if needed

    for img_path in images:
        img = cv2.imread(img_path)
        resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)
        output_path = os.path.join(output_dir, os.path.basename(img_path))
        cv2.imwrite(output_path, resized_img)

# Resize high-resolution and low-resolution images
resize_images("./DIV2K/DIV2K/DIV2K_train_HR", "./Images_HR", (256, 256))
resize_images("./DIV2K/DIV2K/DIV2K_train_HR", "./Images_LR", (64, 64))  # Example for scale factor of 4
