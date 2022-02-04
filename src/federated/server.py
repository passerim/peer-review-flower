import argparse

import flwr as fl

from ..centralized.centralized import Net
from ..utils.utils import set_seed, get_parameters


SEED = 0


def setup_server(port: int, logging_file: str = None):
    
    set_seed(SEED)

    params = get_parameters(Net())

    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_eval=1.0,
        min_fit_clients=2,
        min_eval_clients=2,
        min_available_clients=2,
        initial_parameters=fl.common.weights_to_parameters(params),
    )

    if logging_file:
        fl.common.logger.configure("server", filename=logging_file)
    # Start server
    fl.server.start_server(
        server_address=f"localhost:{port}",
        config={"num_rounds": 1},
        strategy=strategy,
    )


if __name__ == "__main__":

    # Parse command line argument `partition`
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, choices=range(0, 65535), required=True)
    args = parser.parse_args()
