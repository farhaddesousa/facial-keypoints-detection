import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
# Import CNNModel from facial-keypoints-model.py
from facial_keypoints_model import CNNModel, FacialDataset

# Initialize the dataset and DataLoader
test_set = '../dataset/test.csv'
dataset = FacialDataset(test_set, train=False)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Load the model
model = CNNModel()
model.load_state_dict(torch.load('../saved_models_predictions_data/model_masked_rotated_training_data.pth', map_location=torch.device('cpu')))
model.eval()

# Function to predict keypoints
def predict_keypoints(model, img_tensor):
    with torch.no_grad():
        output = model(img_tensor)
    keypoints = output.numpy().reshape(-1, 2)
    return keypoints

# Lists to store images and predictions
all_images = []
all_predictions = []

# Iterate over the DataLoader and store predictions
for idx, images in enumerate(dataloader):
    img_tensor = images  # Shape: [1, 1, 96, 96]
    keypoints = predict_keypoints(model, img_tensor)

    # Store the image and predictions
    all_images.append(images.numpy().squeeze())
    all_predictions.append(keypoints)

    # Optionally, print progress every 1000 images
    if idx % 1000 == 0:
        print(f"Processed {idx} images")

all_predictions_array = np.array(all_predictions)  # Shape: (num_samples, 15, 2)
all_images_array = np.array(all_images)

# Save to a NumPy binary file
np.save('../saved_models_predictions_data/predictions_masked_rotated_constrained15.npy', all_predictions_array)
np.save('../saved_models_predictions_data/masked_rotated_15_images.npy', all_images_array)




