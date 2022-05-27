import os

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import nn
from torchvision.datasets import CIFAR100


def load_data(src_path: str = "."):
    """Load CIFAR-100 (training and test set)."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4, padding_mode="edge"),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
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
        criterion = torch.nn.CrossEntropyLoss(reduction="sum")
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


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def _weights_init(m):
    pass


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option="B"):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == "A":
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(
                        x[:, :, ::2, ::2],
                        (0, 0, 0, 0, planes // 4, planes // 4),
                        "constant",
                        0,
                    )
                )
            elif option == "B":
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.expansion * planes),
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100, **kwargs):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(
            block, 16, num_blocks[0], stride=1, kwargs=kwargs
        )
        self.layer2 = self._make_layer(
            block, 32, num_blocks[1], stride=2, kwargs=kwargs
        )
        self.layer3 = self._make_layer(
            block, 64, num_blocks[2], stride=2, kwargs=kwargs
        )
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, **kwargs):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, kwargs))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20(num_classes=100, option="B"):
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes, option=option)
