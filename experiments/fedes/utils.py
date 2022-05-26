import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100


def load_data(src_path: str = "."):
    """Load CIFAR-100 (training and test set)."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = CIFAR100(
        os.path.join(os.path.abspath(src_path), "data", "cifar100"),
        train=True,
        download=True,
        transform=transform,
    )
    testset = CIFAR100(
        os.path.join(os.path.abspath(src_path), "data", "cifar100"),
        train=False,
        download=True,
        transform=transform,
    )
    return trainset, testset


def train(net, trainloader, criterion=None, optimizer=None, device: str = "cpu"):
    """Train the network on the training set."""
    net.to(device)  # move model to GPU if available
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss(reduction="sum").to(device)
    if optimizer is None:
        optimizer = torch.optim.SGD(
            net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4
        )
    net.train()
    correct, total, epoch_loss = 0, 0, 0.0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    epoch_loss /= total
    accuracy = correct / total
    net.to("cpu")  # move model back to CPU
    return epoch_loss, accuracy


def test(net, testloader, device: str = "cpu"):
    """Validate the network on the entire test set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for _, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= total
    accuracy = correct / total
    net.to("cpu")  # move model back to CPU
    return loss, accuracy


def load_model(entrypoint: str = "mobilenet_v2", finetune: bool = True, **kwargs):
    """Loads pretrained model from torch hub.
    Replaces final classifying layer if classes is specified.
    Args:
        entrypoint: PyTorch Hub model to download.
        classes: Number of classes in final classifying layer.
                 Leave as None to get the downloaded model untouched.
    Returns:
        Loaded model from specified entrypoint.
    """
    model = torch.hub.load("pytorch/vision:v0.12.0", entrypoint, kwargs=kwargs)
    if not finetune:
        for param in model.parameters():
            param.requires_grad = False
    return model
