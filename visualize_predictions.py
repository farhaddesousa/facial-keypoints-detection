import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw

import numpy as np
from PIL import Image, ImageDraw


import numpy as np
import os
from PIL import Image, ImageDraw

# Function to create an image with keypoints
def create_image_with_keypoints(img_array, keypoints):
    img_array = img_array.reshape(96, 96)
    img_array = (img_array * 255.0).astype(np.uint8)  # Convert back to original scale
    img = Image.fromarray(img_array, mode='L').convert('RGB')
    draw = ImageDraw.Draw(img)
    for x, y in keypoints:
        r = 1
        draw.ellipse((x - r, y - r, x + r, y + r), fill='red')
    return img

# Function to visualize an image and its predicted keypoints
def visualize_image(index, all_images, all_predictions, output_dir):
    img_array = all_images[index]  # Get the image by index
    keypoints = all_predictions[index]  # Get the corresponding keypoints by index
    img_with_keypoints = create_image_with_keypoints(img_array, keypoints)  # Create the image with keypoints
    img_with_keypoints.save(f"{output_dir}/predicted_keypoints_{index}.jpg")  # Save the image

# Create folder 'test_images_with_keypoints' if it doesn't exist
output_dir = 'test_images_with_keypoints'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the numpy arrays
all_predictions = np.load('predictions.npy')  # Shape: (1783, 15, 2)
all_images = np.load('images.npy')  # Shape: (1783, 96, 96)

# Loop through all 1783 images and save them with keypoints
for i in range(all_images.shape[0]):
    visualize_image(i, all_images, all_predictions, output_dir)
    if i % 100 == 0:
        print(f"Saved {i} images...")  # Print progress every 100 images

print("Finished saving all images.")
