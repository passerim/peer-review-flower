import argparse

import flwr as fl
from flwr.common import logger, ndarrays_to_parameters
from flwr.server import ServerConfig
from flwr.server.strategy import FedAvg

from examples.centralized.centralized import Net
from examples.centralized.utils import get_parameters, set_seed


def setup_server(port: int, num_rounds=1, logging_file: str = None, seed: int = 0):
    set_seed(seed)

    # Define strategy
    strategy = FedAvg(
        initial_parameters=ndarrays_to_parameters(get_parameters(Net())),
    )

    # Set up logging if a log file is specified
    if logging_file:
        logger.configure("server", filename=logging_file)

    # Start server
    fl.server.start_server(
        server_address=f"localhost:{port}",
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port",
        type=int,
        choices=range(0, 65535),
        required=True,
        help="port used by the server",
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        choices=range(0, 65535),
        default=1,
        help="number of iteration of federated learning",
    )
    parser.add_argument("--seed", default=0, type=int, help="random seed")
    args = parser.parse_args()
    setup_server(args.port, args.num_rounds, seed=args.seed)


if __name__ == "__main__":
    main()
