import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from examples.centralized.utils import set_seed


class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(net, trainloader, epochs: int, verbose=False):
    """Train the network on the training set."""
    net.train()
    device = next(net.parameters()).device
    optimizer = torch.optim.Adam(net.parameters())
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        if verbose:
            print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")


def test(net, testloader):
    """Evaluate the network on the entire test set."""
    net.eval()
    device = next(net.parameters()).device
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= total
    accuracy = correct / total
    return loss, accuracy


def load_data(data_path: str):
    """Load CIFAR-10 (training and test set)."""

    # Normalizes pixels value between -1 and +1
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = CIFAR10(data_path, train=True, download=True, transform=transform)
    testset = CIFAR10(data_path, train=False, download=True, transform=transform)
    return trainset, testset


def main(
    data_path: str = "./data/cifar10",
    epochs: int = 1,
    batch_size: int = 50,
    device: str = "cpu",
    seed: int = 0,
):
    set_seed(seed)

    # Load model
    net = Net().to(device)

    # Load data
    trainset, testset = load_data(data_path)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size)

    # Start centralized training
    train(net, trainloader, epochs=epochs, verbose=True)
    loss, accuracy = test(net, testloader)
    return loss, accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CIFAR-10 centralized training with Pytorch."
    )
    parser.add_argument(
        "--data_path",
        default="./data/cifar10",
        type=str,
        help="path where the cifar-10 dataset is stored",
    )
    parser.add_argument(
        "--epochs", default=1, type=int, help="number of total epochs to run"
    )
    parser.add_argument(
        "--batch_size",
        default=50,
        type=int,
        help="number of images to use for computing gradients",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to use for training (Use: cuda or cpu, Default: cpu)",
    )
    parser.add_argument("--seed", default=0, type=int, help="random seed")
    args = parser.parse_args()
    loss, accuracy = main(
        args.data_path, args.epochs, args.batch_size, args.device, args.seed
    )
    print(f"Final test set performance: loss {loss}, accuracy {accuracy}")
