import argparse

import flwr as fl
from torch import nn
from torch.utils.data import DataLoader

from examples.centralized.centralized import test, train
from examples.centralized.utils import get_parameters, set_parameters
from examples.federated.utils import client_fn
from prflwr.peer_review import PeerReviewNumPyClient, PrConfig


class CifarClient(PeerReviewNumPyClient):
    def __init__(
        self, model: nn.Module, trainloader: DataLoader, testloader: DataLoader
    ):
        self.model = model
        self.device = next(model.parameters()).device
        self.trainloader = trainloader
        self.testloader = testloader

    def get_parameters(self, config):
        return get_parameters(self.model)

    def train(self, parameters, config):
        set_parameters(self.model, parameters)
        epochs = 1
        if config.get("num_epochs") and isinstance(config.get("num_epochs"), int):
            epochs = config.get("num_epochs")
        train(self.model, self.trainloader, epochs=epochs, device=self.device)
        return get_parameters(self.model), len(self.trainloader.dataset), {}

    def review(self, parameters, config):
        set_parameters(self.model, parameters)
        loss, _ = test(self.model, self.testloader, device=self.device)
        return (
            get_parameters(self.model),
            len(self.testloader.dataset),
            {PrConfig.REVIEW_SCORE: float(loss)},
        )

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        loss, accuracy = test(self.model, self.testloader, device=self.device)
        return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}


def setup_client(
    port: int,
    cid: int,
    num_clients: int,
    train_test_fraction: float = 1.0,
    data_path: str = "./data/cifar10",
    batch_size: int = 50,
    device: str = "cpu",
    seed: int = 0,
):
    # Create client
    client = client_fn(
        cid,
        num_clients,
        CifarClient,
        train_test_fraction,
        data_path,
        batch_size,
        device,
        seed,
    )

    # Start client
    fl.client.start_numpy_client(
        server_address=f"localhost:{port}",
        client=client,
    )


if __name__ == "__main__":
    """Create model, load data, define Flower client, start Flower client."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, choices=range(0, 65535), required=True)
    parser.add_argument("--cid", type=int, choices=range(0, 999), required=True)
    parser.add_argument(
        "--num_clients", type=int, choices=range(1, 1000), required=True
    )
    parser.add_argument(
        "--train_test_fraction", type=float, choices=range(0, 1), default=1
    )
    parser.add_argument(
        "--data_path",
        default="./data/cifar10",
        type=str,
        help="path where cifar-10 dataset is stored",
    )
    parser.add_argument(
        "--epochs", default=1, type=int, help="number of total epochs to run"
    )
    parser.add_argument(
        "--batch_size",
        default=50,
        type=int,
        help="number of images to use to compute gradients",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device (Use: cuda or cpu, Default: cpu)",
    )
    parser.add_argument("--seed", default=0, type=int, help="random seed")
    args = parser.parse_args()
    if args.cid >= args.num_clients:
        print(
            "The client id should be an integer lower than the total number of clients."
        )
        exit()
    setup_client(
        args.port,
        args.cid,
        args.num_clients,
        args.train_test_fraction,
        args.data_path,
        args.batch_size,
        args.device,
        args.seed,
    )
