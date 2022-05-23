import argparse

import flwr as fl
from flwr.server.client_manager import SimpleClientManager
from prflwr.peer_reviewed.prserver import PeerReviewServer
from prflwr.utils.pytorch import get_parameters, set_seed

from ..centralized.centralized import Net
from .strategy import PeerReviewedFedAvg

SEED = 0


def setup_server(port: int, num_rounds=1, logging_file: str = None):
    set_seed(SEED)
    params = get_parameters(Net())

    # Define strategy
    strategy = PeerReviewedFedAvg(
        fraction_fit=1.0,
        fraction_review=1.0,
        fraction_eval=1.0,
        min_fit_clients=2,
        min_review_clients=2,
        min_eval_clients=2,
        min_available_clients=2,
        initial_parameters=fl.common.weights_to_parameters(params),
    )

    # Set up logging if a log file is specified
    if logging_file:
        fl.common.logger.configure("server", filename=logging_file)

    # Start server
    fl.server.start_server(
        server_address=f"localhost:{port}",
        server=PeerReviewServer(
            client_manager=SimpleClientManager(), strategy=strategy
        ),
        config={"num_rounds": num_rounds},
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, choices=range(0, 65535), required=True)
    parser.add_argument("--num_rounds", type=int, choices=range(0, 65535), default=1)
    args = parser.parse_args()
    setup_server(args.port, args.num_rounds)


if __name__ == "__main__":
    main()
