import os
import time

import torch
import torchvision.transforms as transforms
from torchvision import datasets
from tqdm import tqdm


def load_data(src_path: str = ".", dataset: str = "CIFAR100"):
    """Load CIFAR-100 (training and test set)."""
    transform_train = transforms.Compose(
        [
            transforms.Pad(4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    transform_test = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = getattr(datasets, dataset)(
        os.path.join(os.path.abspath(src_path), "data", dataset.lower()),
        train=True,
        download=True,
        transform=transform_train,
    )
    testset = getattr(datasets, dataset)(
        os.path.join(os.path.abspath(src_path), "data", dataset.lower()),
        train=False,
        download=True,
        transform=transform_test,
    )
    return trainset, testset


class Timer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.time = time.time()

    def value(self):
        return time.time() - self.time


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(model, trainloader, criterion, optimizer, device="cpu", display=True):
    """Train the network on the training set."""
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    timer = Timer()
    loader = tqdm(trainloader, dynamic_ncols=True)
    model.train()
    for _, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item())
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == targets).sum().item()
        acc_meter.update(correct, targets.size(0))
        if display:
            loader.set_postfix(loss=loss_meter.val)
    return loss_meter.avg, acc_meter.avg, timer.value()


def test(model, testloader, criterion, device):
    """Validate the network on the entire test set."""
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    timer = Timer()
    model.eval()
    with torch.no_grad():
        for _, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss_meter.update(loss.item())
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == targets).sum().item()
            acc_meter.update(correct, targets.size(0))
    return loss_meter.avg, acc_meter.avg, timer.value()


def print_tensor_dict(params):
    kmax = max(len(key) for key in params.keys())
    for i, (key, v) in enumerate(params.items()):
        if key.find("num") < 0:
            print(
                str(i).ljust(5),
                key.ljust(kmax + 3),
                str(tuple(v.shape)).ljust(23),
                torch.typename(v),
                v.requires_grad,
            )
