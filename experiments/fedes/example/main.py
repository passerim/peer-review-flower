import argparse
from collections import OrderedDict
from typing import Callable, Dict, Optional, Tuple

import flwr as fl
import numpy as np
import ray
import torch
import torchvision
from flwr.common.typing import Scalar
from prflwr.utils.flwr import import_dataset_utils
from torch.utils.data import DataLoader, Subset

from .utils import Net, load_data, test, train


import_dataset_utils()  # make Flower dataset utils visible

parser = argparse.ArgumentParser(description="Flower Simulation with PyTorch")
parser.add_argument("--num_client_cpus", type=int, default=1)
parser.add_argument("--num_rounds", type=int, default=10)


# Flower client that will be spawned by Ray
# Adapted from Pytorch quickstart example
class CifarRayClient(fl.client.NumPyClient):
    def __init__(self, cid: str, trainset: Subset):
        self.cid = cid
        self.trainset = trainset
        self.properties: Dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}
        self.net = Net()  # instantiate model
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )  # determine device

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    # def get_properties(self, ins: PropertiesIns) -> PropertiesRes:
    def get_properties(self, ins):
        return self.properties

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict(
            {k: torch.from_numpy(np.copy(v)) for k, v in params_dict}
        )
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):

        # print(f"fit() on client cid={self.cid}")
        self.set_parameters(parameters)

        # load data for this client and get trainloader
        num_workers = len(ray.worker.get_resource_ids()["CPU"])
        trainloader = DataLoader(
            self.trainset,
            shuffle=True,
            batch_size=int(config["batch_size"]),
            workers=num_workers,
        )

        # send model to device
        self.net.to(self.device)

        # train
        train(self.net, trainloader, epochs=int(config["epochs"]), device=self.device)

        # return local model and statistics
        return self.get_parameters(), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):

        # print(f"evaluate() on client cid={self.cid}")
        self.set_parameters(parameters)

        # load data for this client and get trainloader
        num_workers = len(ray.worker.get_resource_ids()["CPU"])
        valloader = DataLoader(
            self.trainset,
            shuffle=False,
            batch_size=int(config["batch_size"]),
            workers=num_workers,
        )

        # send model to device
        self.net.to(self.device)

        # evaluate
        loss, accuracy = test(self.net, valloader, device=self.device)

        # return statistics
        return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)}


def fit_config(rnd: int) -> Dict[str, str]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epoch_global": str(rnd),
        "epochs": str(5),  # number of local epochs
        "batch_size": str(64),
    }
    return config


def set_weights(model: torch.nn.ModuleList, weights: fl.common.Weights) -> None:
    """Set model weights from a list of NumPy ndarrays."""
    state_dict = OrderedDict(
        {
            k: torch.tensor(np.atleast_1d(v))
            for k, v in zip(model.state_dict().keys(), weights)
        }
    )
    model.load_state_dict(state_dict, strict=True)


def get_eval_fn(
    testset: torchvision.datasets.CIFAR10,
) -> Callable[[fl.common.Weights], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, float]]:
        """Use the entire CIFAR-10 test set for evaluation."""

        # determine device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = Net()
        set_weights(model, weights)
        model.to(device)

        testloader = torch.utils.data.DataLoader(testset, batch_size=50)
        loss, accuracy = test(model, testloader, device=device)

        # return statistics
        return loss, {"accuracy": accuracy}

    return evaluate


# Start Ray simulation (a _default server_ will be created)
# This example does:
# 1. Downloads CIFAR-10
# 2. Partitions the dataset into N splits, where N is the total number of
#    clients. We refere to this as `pool_size`. The partition can be IID or non-IID
# 3. Starts a Ray-based simulation where a % of clients are sample each round.
# 4. After the M rounds end, the global model is evaluated on the entire testset.
#    Also, the global model is evaluated on the valset partition residing in each
#    client. This is useful to get a sense on how well the global model can generalise
#    to each client's data.
if __name__ == "__main__":

    # parse input arguments
    args = parser.parse_args()

    pool_size = 10  # number of dataset partions (= number of total clients)
    client_resources = {
        "num_cpus": args.num_client_cpus
    }  # each client will get allocated 1 CPUs

    # download CIFAR10 dataset
    trainset, testset = load_data(dataset="CIFAR10")

    # partition dataset (use a large `alpha` to make it IID;
    # a small value (e.g. 1) will make it non-IID)
    # This will create a new directory called "federated": in the directory where
    # CIFAR-10 lives. Inside it, there will be N=pool_size sub-directories each with
    # its own train/set split.
    targets = np.array(trainset.targets)
    idxs = np.array(range(len(targets)))
    dataset = [idxs, targets]
    train_partitions, _ = fl.dataset.utils.common.create_lda_partitions(
        dataset, num_partitions=pool_size, concentration=0.1, accept_imbalanced=True
    )

    # configure the strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.1,
        min_fit_clients=10,
        min_available_clients=pool_size,  # All clients should be available
        on_fit_config_fn=fit_config,
        eval_fn=get_eval_fn(testset),  # centralised testset evaluation of global model
    )

    def client_fn(cid: str):
        cid = int(cid)
        trainsubset = Subset(trainset, train_partitions[cid])
        return CifarRayClient(str(cid), trainsubset)

    # (optional) specify ray config
    ray_config = {"include_dashboard": False}

    print(Subset(trainset, train_partitions[0]).indices)

    # start simulation
    """ fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=pool_size,
        client_resources=client_resources,
        num_rounds=args.num_rounds,
        strategy=strategy,
        ray_init_args=ray_config,
    ) """
