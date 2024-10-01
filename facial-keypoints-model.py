import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
print("Hello world")

class FacialDataset(Dataset):
    """Facial  Keypoints dataset."""

    def __init__(self, csv_file, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file.
            transform(callable, optimal): Optional transform to be applied
                on a sample.
        """
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]

        labels = torch.tensor(sample[:30].values.astype(np.float32))
        image = torch.tensor(list(map(int, sample[30].split())), dtype=torch.float32).view(1, 96, 96)  # Reshaping to 1x96x96

        if self.transform:
            image = self.transform(image)

        return image, labels

transform = torch.tensor

train_csv_file_path = 'dataset/training.csv'
train_dataset = FacialDataset(csv_file=train_csv_file_path, transform=transform)

#testing if FacialDataset works
image, labels = train_dataset.__getitem__(5)
print("Hello world")
print(image)
print(labels)



# Defining Train and Test DataLoaders
batch_size = 64
split_data = torch.utils.data.random_split(train_dataset, [0.8,0.2])
train_data = split_data[0]
validation_data = split_data[1]
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)



device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


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
#print(model)

#Defining loss function
class EuclideanLoss(nn.Module):
    def __init__(self):
        super(EuclideanLoss, self).__init__()


    def forward(self, predicted, actual):
        # Reshape predicted and actual if they are not already in the shape [batch_size, num_keypoints, 2]
        predicted = predicted.view(-1, 15, 2)  # Assuming output is [batch_size, 30]
        actual = actual.view(-1, 15, 2)

        # Compute the Euclidean distance (L2 norm) squared between each pair of corresponding keypoints
        return torch.mean(torch.norm(predicted - actual, dim=2, p=2) ** 2)


loss_fn = EuclideanLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# for idx, (images, keypoints) in enumerate(train_dataloader):
#     print(idx)
#     #print(images)
#     print(keypoints)
#     #break  # remove or adjust as needed
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for idx, (images, keypoints) in enumerate(dataloader):
        images, keypoints = images.to(device).float(), keypoints.to(device)

        # Compute prediction error
        pred = model(images)
        loss = loss_fn(pred, keypoints)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if idx % 100 == 0:
            loss, current = loss.item(), (idx + 1) * len(images)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# def test(dataloader, model, loss_fn):
#     size = len(dataloader.dataset)
#     num_batches = len(dataloader)
#     model.eval()
#     test_loss, correct = 0, 0
#     with torch.no_grad():
#         for X, y in dataloader:
#             X, y = X.to(device).float(), y.to(device)
#             pred = model(X)
#             test_loss += loss_fn(pred, y).item()
#             correct += (pred.argmax(1) == y).type(torch.float).sum().item()
#     test_loss /= num_batches
#     correct /= size
#     print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


#Train and test model:
# epochs = 5
# for t in range(epochs):
#     print(f"Epoch {t+1}\n-------------------------------")
#     train(train_dataloader, model, loss_fn, optimizer)
#     #test(test_dataloader, model, loss_fn)
# print("Training and Testing complete")


# Set print options to increase the threshold for printing
# torch.set_printoptions(threshold=10_000)






