import argparse
import json
import os
import time

import torch
import torch.utils.data
from torch import nn
from torch.backends import cudnn

from resnet import WideResNet
from utils import (
    load_data,
    print_tensor_dict,
    test,
    train,
    seed_all,
    create_iterator,
    create_optimizer
)

cudnn.benchmark = True

parser = argparse.ArgumentParser(
    description="Training a Wide Residual Network on CIFAR-10 and CIFAR-100 datasets."
)
# Model options
parser.add_argument("--model", default="resnet", type=str)
parser.add_argument("--depth", default=16, type=int)
parser.add_argument("--width", default=4, type=float)
parser.add_argument("--dataset", default="CIFAR100", type=str)
parser.add_argument("--dataroot", default=".", type=str)
parser.add_argument("--dtype", default="float", type=str)
parser.add_argument("--nthread", default=1, type=int)
parser.add_argument("--seed", default=1, type=int)
# Training options
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--lr", default=0.1, type=float)
parser.add_argument(
    "--epochs", default=200, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument("--weight_decay", default=0.0005, type=float)
parser.add_argument(
    "--epoch_step",
    default="[60,120,160]",
    type=str,
    help="json list with epochs to drop lr on",
)
parser.add_argument("--lr_decay_ratio", default=0.2, type=float)
parser.add_argument("--resume", default="", type=str)
# Device options
parser.add_argument("--cuda", default=torch.cuda.is_available(), type=bool)
parser.add_argument(
    "--save",
    default="./logs/" + time.ctime().replace(" ", "_"),
    type=str,
    help="save parameters and logs in this folder",
)


def log(model, epoch, optimizer, metrics, opt):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        os.path.join(opt.save, "model.pt"),
    )
    z = {**metrics}
    with open(os.path.join(opt.save, "log.txt"), "a") as flog:
        flog.write("json_stats: " + json.dumps(z) + "\n")
    print(z)


def main():
    opt = parser.parse_args()
    print("parsed options:", vars(opt))
    device = "cuda" if (opt.cuda and torch.cuda.is_available()) else "cpu"

    seed_all(opt.seed)

    num_classes = 10 if opt.dataset == "CIFAR10" else 100
    trainset, testset = load_data(dataset=opt.dataset)
    trainloader = create_iterator(trainset, True, opt)
    testloader = create_iterator(testset, False, opt)

    model = WideResNet(opt.depth, opt.width, num_classes)

    criterion = nn.CrossEntropyLoss()

    epoch_step = json.loads(opt.epoch_step)
    optimizer = create_optimizer(model, opt)

    epoch = 0
    if opt.resume != "":
        state_dict = torch.load(opt.resume, map_location=device)
        epoch = state_dict["epoch"]
        model.load_state_dict(state_dict["model_state_dict"])
        optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    model.to(device)
    criterion.to(device)

    maxepoch = opt.epochs
    assert epoch < maxepoch and maxepoch > 0

    print("\nParameters:")
    print_tensor_dict(model.state_dict())

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\nTotal number of parameters:", n_parameters)

    if not os.path.exists(opt.save):
        os.makedirs(opt.save)

    while epoch <= maxepoch:
        epoch += 1
        if epoch in epoch_step:
            lr = optimizer.param_groups[0]["lr"]
            optimizer = create_optimizer(opt, lr * opt.lr_decay_ratio)
        train_loss, train_acc, train_time = train(
            model, trainloader, criterion, optimizer, device
        )
        test_loss, test_acc, test_time = test(model, testloader, criterion, device)
        print(
            log(
                model,
                epoch,
                optimizer,
                {
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "test_loss": test_loss,
                    "test_acc": test_acc,
                    "epoch": epoch,
                    "num_classes": num_classes,
                    "n_parameters": n_parameters,
                    "train_time": train_time,
                    "test_time": test_time,
                },
                opt,
            )
        )
        print(
            "==> id: %s (%d/%d), test_acc: \33[91m%.2f\033[0m"
            % (opt.save, epoch, opt.epochs, test_acc)
        )


if __name__ == "__main__":
    main()
