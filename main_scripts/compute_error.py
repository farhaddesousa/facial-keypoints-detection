import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from facial_keypoints_model import CNNModel, FacialDataset

# Initialize the dataset and DataLoader
train_set = 'dataset/training.csv'  # Path to your training dataset
dataset = FacialDataset(train_set)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# Load the model
model = CNNModel()
model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))
model.eval()


# Function to predict keypoints
def predict_keypoints(model, img_tensor):
    with torch.no_grad():
        output = model(img_tensor)
    keypoints = output.numpy().reshape(-1, 2)  # Reshape to (15, 2)
    return keypoints


# RMSE calculation
def calculate_rmse(predictions, ground_truths):
    """
    Calculate the RMSE for all keypoints (30 values per image: 15 x, y pairs).

    Args:
        predictions (np.ndarray): Predicted keypoints, shape (num_images, 15, 2)
        ground_truths (np.ndarray): Ground truth keypoints, shape (num_images, 15, 2)

    Returns:
        float: RMSE over all keypoints.
    """
    # Flatten the predictions and ground truths to shape (num_images, 30)
    predictions_flat = predictions.reshape(-1, 30)
    ground_truths_flat = ground_truths.reshape(-1, 30)

    # Compute the mean squared error for each coordinate
    mse = np.mean((predictions_flat - ground_truths_flat) ** 2)

    # Take the square root to get RMSE
    rmse = np.sqrt(mse)
    return rmse


# Lists to store predictions and ground truth labels
all_predictions = []
all_ground_truths = []

# Iterate over the DataLoader and store predictions and actual labels
for idx, (images, labels) in enumerate(dataloader):
    img_tensor = images  # Shape: [1, 1, 96, 96]
    keypoints = predict_keypoints(model, img_tensor)  # Predict keypoints

    # Append predictions and actual labels (reshaped to (15, 2))
    all_predictions.append(keypoints)
    all_ground_truths.append(labels.numpy().reshape(15, 2))  # Ground truth keypoints

    # Optionally, print progress every 1000 images
    if idx % 1000 == 0:
        print(f"Processed {idx} images")

# Convert the lists to NumPy arrays for easier processing
all_predictions_array = np.array(all_predictions)  # Shape: (num_images, 15, 2)
all_ground_truths_array = np.array(all_ground_truths)  # Shape: (num_images, 15, 2)

# Calculate RMSE
rmse = calculate_rmse(all_predictions_array, all_ground_truths_array)
print(f"RMSE for all keypoints: {rmse:.4f}")
