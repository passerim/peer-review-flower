import argparse
from typing import Callable, Dict, Optional, Tuple

import flwr as fl
import numpy as np
from flwr.common.parameter import weights_to_parameters
from prflwr.utils.pytorch import set_parameters
from prflwr.utils.flwr import import_dataset_utils
from torch.utils.data import DataLoader, Dataset, Subset

from ..utils import load_data, load_efficientnet, test
from .client import CifarClient

import_dataset_utils()  # make Flower dataset utils visible

SEED = 0
CLIENT_RESOURCES = 1  # each client will get allocated 1 CPUs

parser = argparse.ArgumentParser(description="Flower Simulation with PyTorch")
parser.add_argument("--pool_size", type=int, default=100)
parser.add_argument("--num_rounds", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--fraction_fit", type=float, default=1.0)
parser.add_argument("--local_epochs", type=int, default=5)


def get_eval_fn(
    testset: Dataset,
) -> Callable[[fl.common.Weights], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, float]]:
        """Use the entire CIFAR-100 test set for evaluation."""

        model = load_efficientnet(finetune=False, classes=100)
        set_parameters(model, weights_to_parameters(weights))

        testloader = DataLoader(testset, batch_size=32)
        loss, accuracy = test(model, testloader)

        # return statistics
        return loss, {"accuracy": accuracy}

    return evaluate


def main():
    args = parser.parse_args()  # parse input arguments
    pool_size = args.pool_size  # number of dataset partions (= number of total clients)
    num_rounds = args.num_rounds
    batch_size = args.batch_size
    fraction_fit = args.fraction_fit
    local_epochs = args.local_epochs

    # Download CIFAR100 dataset
    trainset, testset = load_data()

    # Partition dataset (use a large `alpha` to make it IID, a small value (e.g. 1)
    # will make it non-IID).
    # This will create a new directory called "federated": in the directory where
    # CIFAR-100 lives. Inside it, there will be N=pool_size sub-directories each with
    # its own train/set split.
    targets = np.array(trainset.targets)
    idxs = np.array(range(len(targets)))
    dataset = [idxs, targets]

    partitions, _ = fl.dataset.utils.common.create_lda_partitions(
        dataset, num_partitions=pool_size, concentration=0.1, accept_imbalanced=True
    )

    def fit_config(rnd: int) -> Dict[str, str]:
        """Return a configuration with static batch size and (local) epochs."""
        config = {
            "epoch_global": str(rnd),
            "epochs": str(local_epochs),  # number of local epochs
            "batch_size": str(batch_size),
        }
        return config

    # Configure the strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=fraction_fit,
        min_fit_clients=int(fraction_fit * pool_size),
        min_available_clients=pool_size,  # all clients should be available
        on_fit_config_fn=fit_config,
        eval_fn=get_eval_fn(testset),  # centralised testset evaluation of global model
    )

    def client_fn(cid: str):
        # create a single client instance
        cid = int(cid)
        traisubset = Subset(trainset, partitions[cid])
        testsubset = Subset(trainset, partitions[cid])
        return CifarClient(traisubset, testsubset, cid=cid)

    # Set up logging if a log file is specified
    fl.common.logger.configure("fedavg_sim", filename="fedavg_sim.log")

    # Start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=pool_size,
        client_resources=CLIENT_RESOURCES,
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
