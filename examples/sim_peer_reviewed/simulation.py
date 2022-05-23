import argparse
from functools import partial

import flwr as fl
from flwr.server.client_manager import SimpleClientManager
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from ..centralized.centralized import Net, load_data
from prflwr.utils.pytorch import set_seed, get_parameters
from ..peer_reviewed.strategy import PeerReviewedFedAvg
from prflwr.peer_reviewed.prserver import PeerReviewServer
from ..peer_reviewed.client import CifarClient
from prflwr.simulation.app import start_simulation


SEED = 0
BATCH_SIZE = 32
DEVICE = "cpu"


def client_fn(cid: str, num_clients: int) -> fl.client.NumPyClient:
    set_seed(SEED)

    # Load model
    net = Net().to(DEVICE)

    # Load data
    trainset, testset, _ = load_data()
    trainset_sampler = DistributedSampler(
        trainset, num_replicas=num_clients, rank=int(cid), shuffle=True, seed=SEED
    )
    trainloader = DataLoader(trainset, sampler=trainset_sampler, batch_size=BATCH_SIZE)
    testset_sampler = DistributedSampler(
        testset, num_replicas=num_clients, rank=int(cid), shuffle=True, seed=SEED
    )
    testloader = DataLoader(testset, sampler=testset_sampler, batch_size=BATCH_SIZE)
    return CifarClient(net, trainloader, testloader)


def setup_server(num_rounds: int = 1, num_clients: int = 2, logging_file: str = None):
    set_seed(SEED)
    params = get_parameters(Net())

    # Define strategy
    strategy = PeerReviewedFedAvg(
        fraction_review=1.0,
        fraction_fit=1.0,
        fraction_eval=1.0,
        min_fit_clients=2,
        min_eval_clients=2,
        min_available_clients=2,
        initial_parameters=fl.common.weights_to_parameters(params),
    )

    # Set up logging if a log file is specified
    if logging_file:
        fl.common.logger.configure("server", filename=logging_file)

    # Start simulation
    start_simulation(
        client_fn=partial(client_fn, num_clients=num_clients),
        num_clients=num_clients,
        num_rounds=num_rounds,
        server=PeerReviewServer(
            client_manager=SimpleClientManager(), strategy=strategy
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_rounds", type=int, choices=range(0, 65535), default=1)
    parser.add_argument("--num_clients", type=int, choices=range(0, 65535), default=2)
    args = parser.parse_args()
    setup_server(args.num_rounds, args.num_clients)


if __name__ == "__main__":
    main()
