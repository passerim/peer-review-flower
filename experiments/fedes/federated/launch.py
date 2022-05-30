import argparse
import json
from typing import Callable, Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import torch
from flwr.common.parameter import weights_to_parameters
from prflwr.utils.flwr import import_dataset_utils
from prflwr.utils.pytorch import get_parameters, set_weights
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import VisionDataset

from ..resnet import WideResNet
from ..utils import load_data, seed_all, test
from .client import CifarClient

cudnn.benchmark = True
import_dataset_utils()  # make Flower dataset utils visible

parser = argparse.ArgumentParser(
    description="""Training a federated learning Wide Residual Network
    model on CIFAR-10 and CIFAR-100 datasets with Flower."""
)
# Flower federated learning options
parser.add_argument("--pool_size", type=int, default=10)
parser.add_argument("--num_rounds", type=int, default=1)
parser.add_argument("--local_epochs", type=int, default=1)
parser.add_argument("--fraction_fit", type=float, default=0.1)
parser.add_argument("--fraction_eval", type=float, default=0)
# Model options
parser.add_argument("--model", default="resnet", type=str)
parser.add_argument("--depth", default=10, type=int)
parser.add_argument("--width", default=1, type=float)
parser.add_argument("--dataset", default="CIFAR100", type=str)
parser.add_argument("--dataroot", default=".", type=str)
parser.add_argument("--seed", default=1, type=int)
# Training options
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--lr", default=0.1, type=float)
parser.add_argument("--weight_decay", default=0.0005, type=float)
parser.add_argument("--epoch_step", default="[60,120,160]", type=str)
parser.add_argument("--lr_decay_ratio", default=0.2, type=float)
# Device options
parser.add_argument("--cuda", default=False, type=bool)


def get_eval_fn(
    testset: Dataset, model: nn.Module, device: str = "cpu"
) -> Callable[[fl.common.Weights], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, float]]:
        """Use the entire CIFAR-100 test set for evaluation."""
        set_weights(model, weights)
        testloader = DataLoader(testset, batch_size=128)
        criterion = nn.CrossEntropyLoss().to(device)
        loss, accuracy, _ = test(model.to(device), testloader, criterion, device)
        return loss, {"accuracy": accuracy}

    return evaluate


def get_client_function(
    trainset: VisionDataset,
    testset: VisionDataset,
    trainset_partitions: List[np.ndarray],
    testset_partitions: List[np.ndarray],
    device: str = "cpu",
):
    def client_fn(cid: str):
        # create a single client instance
        cid = int(cid)
        trainsubset = Subset(trainset, trainset_partitions[cid])
        testsubset = Subset(testset, testset_partitions[cid])
        print("running")
        return CifarClient(trainsubset, testsubset, device, cid)

    return client_fn


def get_fit_config(
    model_opt: Dict[str, int],
    batch_size: int,
    local_epochs: int,
    base_lr: float,
    lr_steps: List[int] = None,
    lr_decay: float = None,
):
    def fit_config(rnd: int) -> Dict[str, str]:
        """Return a configuration with static batch size and (local) epochs."""
        lr = base_lr
        if lr_steps is not None and lr_decay is not None:
            lr *= lr_decay ** sum([1 if rnd >= e else 0 for e in lr_steps])
        config = {
            "model_opt": model_opt,
            "batch_size": str(batch_size),
            "gloabal_epoch": str(rnd),
            "local_epochs": str(local_epochs),  # number of local epochs
            "lr": lr,
        }
        return config

    return fit_config


def main():
    opt = parser.parse_args()  # parse input arguments
    print("parsed options:", vars(opt))
    pool_size = opt.pool_size  # number of dataset partions (= number of total clients)
    num_rounds = opt.num_rounds
    local_epochs = opt.local_epochs
    fraction_fit = opt.fraction_fit
    batch_size = opt.batch_size
    epoch_step = json.loads(opt.epoch_step)

    seed_all(opt.seed)
    device = "cuda" if (opt.cuda and torch.cuda.is_available()) else "cpu"

    # Download CIFAR datasets
    num_classes = 10 if opt.dataset == "CIFAR10" else 100
    trainset, testset = load_data(dataset=opt.dataset)

    # Partition dataset (use a large `alpha` to make it IID, a small value (e.g. 1)
    # will make it non-IID).
    # This will create a new directory called "federated": in the directory where
    # CIFAR-100 lives. Inside it, there will be N=pool_size sub-directories each with
    # its own train/set split.
    targets = np.array(trainset.targets)
    idxs = np.array(range(len(targets)))
    dataset = [idxs, targets]
    train_partitions, _ = fl.dataset.utils.common.create_lda_partitions(
        dataset, num_partitions=pool_size, concentration=0.1, accept_imbalanced=True
    )
    targets = np.array(testset.targets)
    idxs = np.array(range(len(targets)))
    dataset = [idxs, targets]
    test_partitions, _ = fl.dataset.utils.common.create_lda_partitions(
        dataset, num_partitions=pool_size, concentration=0.1, accept_imbalanced=True
    )

    # Create model
    model = WideResNet(opt.depth, opt.width, num_classes)
    model_weights = get_parameters(model)

    # Configure the strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=fraction_fit,
        fraction_eval=0,
        min_fit_clients=int(fraction_fit * pool_size),
        min_eval_clients=0,
        min_available_clients=pool_size,  # all clients should be available
        eval_fn=get_eval_fn(
            testset, model, device
        ),  # centralised testset evaluation of global model
        on_fit_config_fn=get_fit_config(
            {"depth": opt.depth, "width": opt.width, "num_classes": num_classes},
            batch_size,
            local_epochs,
            opt.lr,
            epoch_step,
        ),
        initial_parameters=weights_to_parameters(model_weights),
    )

    # Set up logging if a log file is specified
    fl.common.logger.configure("fedavg_sim", filename="fedavg_sim.log")

    # Start simulation
    fl.simulation.start_simulation(
        client_fn=get_client_function(
            trainset, testset, train_partitions, test_partitions, device
        ),
        num_clients=pool_size,
        num_rounds=num_rounds,
        strategy=strategy,
        ray_init_args={"include_dashboard": False},
    )


# Start Ray simulation (a _default server_ will be created)
# This example does:
# 1. Downloads CIFAR-100
# 2. Partitions the dataset into N splits, where N is the total number of
#    clients. We refere to this as `pool_size`. The partition can be IID or non-IID
# 3. Starts a Ray-based simulation where a % of clients are sample each round.
# 4. After the M rounds end, the global model is evaluated on the entire testset.
#    Also, the global model is evaluated on the valset partition residing in each
#    client. This is useful to get a sense on how well the global model can generalise
#    to each client's data.
if __name__ == "__main__":
    main()
