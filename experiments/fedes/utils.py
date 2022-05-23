import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100


def load_data():
    """Load CIFAR-100 (training and test set)."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = CIFAR100(
        "./data/cifar100", train=True, download=True, transform=transform
    )
    testset = CIFAR100(
        "./data/cifar100", train=False, download=True, transform=transform
    )
    return trainset, testset


def train(net, trainloader, epochs, device: str = "cpu"):
    """Train the network on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4
    )
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
    net.to("cpu")  # move model back to CPU


def test(net, testloader, device: str = "cpu"):
    """Validate the network on the entire test set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss()
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


def load_efficientnet(
    entrypoint: str = "nvidia_efficientnet_b0",
    finetune: bool = True,
    classes: int = None,
):
    """Loads pretrained efficientnet model from torch hub. Replaces final
    classifying layer if classes is specified.
    Args:
        entrypoint: EfficientNet model to download.
                    For supported entrypoints, please refer
                    https://pytorch.org/hub/nvidia_deeplearningexamples_efficientnet/
        classes: Number of classes in final classifying layer. Leave as None to get the downloaded
                 model untouched.
    Returns:
        EfficientNet Model
    Note: One alternative implementation can be found at https://github.com/lukemelas/EfficientNet-PyTorch
    """
    efficientnet = torch.hub.load(
        "NVIDIA/DeepLearningExamples:torchhub", entrypoint, pretrained=True
    )
    if not finetune:
        for param in efficientnet.parameters():
            param.requires_grad = False
    if classes is not None:
        # Replaces the final layer of the classifier.
        num_features = efficientnet.classifier.fc.in_features
        efficientnet.classifier.fc = torch.nn.Linear(num_features, classes)
    return efficientnet
