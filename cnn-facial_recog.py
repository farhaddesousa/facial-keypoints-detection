import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # Define the convolutional layers
        # one layer for greyscale
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # Define the max pooling layers
        self.pool = nn.MaxPool2d(2, 2)
        # Define the fully connected layers
        self.fc1 = nn.Linear(128 * 12 * 12, 512)
        self.fc2 = nn.Linear(512, 30)  # Output layer with 30 classes

    def forward(self, x):
        # Apply convolutional layers followed by max pooling and relu activation
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # Flatten the output for fully connected layers
        x = x.view(-1, 128 * 12 * 12)
        # Apply fully connected layers with relu activation
        x = F.relu(self.fc1(x))
        # Output layer
        x = self.fc2(x)
        return x


# Create an instance of the model
model = CNNModel()

# Print the model architecture
print(model)
