import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from prflwr.utils.pytorch import set_seed


SEED = 0
BATCH_SIZE = 32
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


def train(net, trainloader, epochs: int, device="cpu",  verbose=False):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        if verbose:
            print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")


def test(net, testloader, device="cpu"):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    correct, total, loss = 0, 0, 0.0
    net.eval()
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


def load_data():
    """Load CIFAR-10 (training and test set)."""
    
    # Normalizes pixels value between -1 and +1
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = CIFAR10("./data/cifar10", train=True, download=True, transform=transform)
    testset = CIFAR10("./data/cifar10", train=False, download=True, transform=transform)
    num_examples = {"trainset": len(trainset), "testset": len(testset)}
    return trainset, testset, num_examples


def main():
    set_seed(SEED)

    # Load model
    net = Net().to(DEVICE)

    # Load data
    trainset, testset, _ = load_data()
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)

    # Start centralized training
    train(net, trainloader, epochs=5, device=DEVICE, verbose=True)
    loss, accuracy = test(net, testloader, device=DEVICE)
    print(f"Final test set performance: loss {loss}, accuracy {accuracy}")


if __name__ == "__main__":
    main()
