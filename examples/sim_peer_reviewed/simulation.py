import argparse
from functools import partial

from flwr.common import logger, ndarrays_to_parameters
from flwr.server import ServerConfig
from flwr.server.client_manager import SimpleClientManager
from flwr.simulation import start_simulation

from examples.centralized.centralized import Net
from examples.centralized.utils import get_parameters, set_seed
from examples.federated.client import client_fn
from examples.peer_reviewed.client import CifarClient
from prflwr.peer_review.server import PeerReviewServer
from prflwr.peer_review.strategy import PeerReviewedFedAvg


def setup_server(
    num_rounds: int = 1,
    num_clients: int = 2,
    logging_file: str = None,
    data_path: str = "./data/cifar10",
    batch_size: int = 50,
    device: str = "cpu",
    seed: int = 0,
):
    set_seed(seed)

    # Define strategy
    strategy = PeerReviewedFedAvg(
        initial_parameters=ndarrays_to_parameters(get_parameters(Net())),
    )

    # Set up logging if a log file is specified
    if logging_file:
        logger.configure("server", filename=logging_file)

    # Start simulation
    hist = start_simulation(
        client_fn=partial(
            client_fn,
            num_clients=num_clients,
            client_class=CifarClient,
            data_path=data_path,
            batch_size=batch_size,
            device=device,
            seed=seed,
        ),
        num_clients=num_clients,
        server=PeerReviewServer(
            client_manager=SimpleClientManager(), strategy=strategy
        ),
        config=ServerConfig(num_rounds=num_rounds),
        client_resources={"num_cpus": 1, "num_gpus": 0 if device == "cpu" else 1},
        ray_init_args={"local_mode": True, "include_dashboard": False},
    )
    return hist


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_rounds", type=int, choices=range(0, 65535), default=1)
    parser.add_argument("--num_clients", type=int, choices=range(0, 65535), default=2)
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
    setup_server(
        args.num_rounds,
        args.num_clients,
        None,
        args.data_path,
        args.batch_size,
        args.device,
        args.seed,
    )
