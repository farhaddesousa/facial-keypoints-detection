import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import os  # For creating directories if needed

# Define the FacialDataset class
class FacialDataset(Dataset):
    """Facial Keypoints dataset."""

    def __init__(self, csv_file, transform=None, train=True):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]

        # Extract image data
        image = np.array(sample['Image'].split(), dtype=np.float32).reshape(1, 96, 96)
        image = image / 255.0  # Normalize the image to [0, 1]
        image = torch.tensor(image, dtype=torch.float32)

        if self.train:
            # Extract labels (all columns except 'Image')
            labels = sample.drop('Image').values.astype(np.float32)

            # Create a mask where labels are not NaN
            mask = ~np.isnan(labels)

            # Replace NaNs in labels with zeros (since loss will ignore them)
            labels = np.nan_to_num(labels, nan=0.0)

            # Convert labels and mask to tensors
            labels = torch.tensor(labels, dtype=torch.float32)
            mask = torch.tensor(mask.astype(np.float32), dtype=torch.float32)

            return image, labels, mask
        else:
            # For test data, return only the image
            return image


# Load the dataset
if __name__ == "__main__":
    train_csv_file_path = '../dataset/training.csv'
    train_dataset = FacialDataset(csv_file=train_csv_file_path)

    # Defining Train and Validation DataLoaders
    batch_size = 64
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_data, validation_data = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)

# Detect if GPU is available
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Define the CNN model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Output: 32x96x96
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Output: 64x48x48
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Output: 128x24x24
        # Define the max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # Define the fully connected layers
        self.fc1 = nn.Linear(128 * 12 * 12, 512)
        self.fc2 = nn.Linear(512, 30)  # Output layer with 30 values

    def forward(self, x):
        # Apply convolutional layers with ReLU activation and pooling
        x = self.pool(F.relu(self.conv1(x)))  # Output: 32x48x48
        x = self.pool(F.relu(self.conv2(x)))  # Output: 64x24x24
        x = self.pool(F.relu(self.conv3(x)))  # Output: 128x12x12
        # Flatten the output for fully connected layers
        x = x.view(-1, 128 * 12 * 12)
        # Apply fully connected layers with ReLU activation
        x = F.relu(self.fc1(x))
        # Output layer
        x = self.fc2(x)
        return x

# Create an instance of the model and move it to the device
model = CNNModel().to(device)

# Define the loss function
class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, predicted, actual, mask):
        # predicted, actual, mask are tensors of shape [batch_size, num_outputs]
        # Compute squared differences
        squared_diff = (predicted - actual) ** 2
        # Apply mask
        masked_squared_diff = squared_diff * mask
        # Sum over all elements
        loss = masked_squared_diff.sum()
        # Sum over mask elements
        total_mask = mask.sum()
        # Compute mean loss
        mean_loss = loss / total_mask.clamp(min=1e-8)  # Avoid division by zero
        return mean_loss

loss_fn = MaskedMSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Define the training function
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    total_loss = 0
    for idx, (images, keypoints, mask) in enumerate(dataloader):
        images = images.to(device).float()
        keypoints = keypoints.to(device)
        mask = mask.to(device)

        # Forward pass
        pred = model(images)

        # Compute loss
        loss = loss_fn(pred, keypoints, mask)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if idx % 100 == 0:
            loss_value, current = loss.item(), idx * len(images)
            print(f"loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]")

    average_loss = total_loss / len(dataloader)
    print(f"Training Average Loss: {average_loss:>8f}")

# Define the testing (validation) function
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for images, keypoints, mask in dataloader:
            images = images.to(device).float()
            keypoints = keypoints.to(device)
            mask = mask.to(device)

            pred = model(images)

            # Compute loss
            loss = loss_fn(pred, keypoints, mask)
            test_loss += loss.item()

    average_loss = test_loss / num_batches
    print(f"Validation Average Loss: {average_loss:>8f} \n")
    return average_loss

# Train and validate the model
if __name__ == "__main__":
    epochs = 5
    best_validation_loss = float('inf')  # Initialize the best validation loss
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        validation_loss = test(validation_dataloader, model, loss_fn)

        # Save the model if validation loss has decreased
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            model_path = 'best_model_rotated_training_data.pth'
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path} with validation loss {best_validation_loss:>8f}")

    print("Training complete")
