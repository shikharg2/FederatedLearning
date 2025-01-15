"""FlwrTestTemplate: A Flower / PyTorch app."""

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from torch.utils.data import DataLoader
import pandas as pd

df = pd.read_csv('/Users/shikhar/Desktop/Pytorch/Dataset/Dataset + DDoS/dataset+DDos.csv', low_memory=False)
if df.isnull().sum().sum() > 0:
    df.fillna(0, inplace=True)  # Filling missing values with 0
features = df.iloc[:, :-1].values
labels = df.iloc[:, -1].values

scaler = MinMaxScaler()
features = scaler.fit_transform(features)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

input_dim = X_train.shape[1]

class Net(nn.Module):

    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.fc1 = nn.Linear(64 * (input_dim - 4), 128)
        self.fc2 = nn.Linear(128, 2)  # Binary Classification Layer

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return x


class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]



def load_data(partition_id: int, num_partitions: int):

    # Only initialize `FederatedDataset` once

    df = pd.read_csv('/Users/shikhar/Desktop/Pytorch/Dataset/Dataset + DDoS/dataset+DDos.csv', low_memory=False)
    if df.isnull().sum().sum() > 0:
        df.fillna(0, inplace=True)  # Filling missing values with 0
    features = df.iloc[:, :-1].values
    labels = df.iloc[:, -1].values

    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # DataLoaders for training and testing
    train_dataset = CustomDataset(X_train, y_train)
    test_dataset = CustomDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader


def train(net, trainloader, epochs, device):
    """Train the model on the training set."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for data, labels in trainloader:
            optimizer.zero_grad()
            outputs = net(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    print(f'Epoch {epochs + 1}, Loss: {running_loss / len(trainloader):.4f}')
    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):
    net.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    correct = 0
    total = 0

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():  # Disable gradient computation
        for data, labels in testloader:
            # Move data and labels to the specified device (e.g., GPU)
            data, labels = data.to(device), labels.to(device)

            # Forward pass
            outputs = net(data)

            # Compute loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Compute predictions
            _, predicted = torch.max(outputs.data, 1)

            # Update accuracy
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate average loss and accuracy
    average_loss = total_loss / len(testloader)
    accuracy = 100 * correct / total

    print(f'Test Loss: {average_loss:.4f}, Test Accuracy: {accuracy:.2f}%')
    return average_loss, accuracy


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
