import argparse
import os
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
import numpy as np
from flwr.common import Parameters, Scalar, parameters_to_ndarrays
from flwr.server import ClientManager, ServerConfig
from flwr.server.client_proxy import ClientProxy
from sklearn.metrics import mean_squared_error

from prflwr.peer_review import PeerReviewServer, ReviewIns, ReviewRes, TrainRes
from prflwr.peer_review.strategy import (
    AggregateReviewException,
    AggregateTrainException,
    PeerReviewedFedAvg,
)
from prflwr.peer_review.strategy.exceptions import ConfigureReviewException

# Import utility functions
if os.getenv("RUN_DEV"):
    from .utils import *
else:
    from utils import *


class FederatedGradientBoostingStrategy(PeerReviewedFedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def aggregate_train(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, TrainRes]],
        failures: List[Union[Tuple[ClientProxy, TrainRes], BaseException]],
        parameters: Optional[Parameters] = None,
    ) -> List[Tuple[Optional[Parameters], Dict[str, Scalar]]]:
        if not results:
            raise AggregateTrainException
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            raise AggregateTrainException

        # Save ClientProxies
        self.current_round_clients = [proxy for proxy, _ in results]

        # Collect results
        return [(fit_res.parameters, {}) for _, fit_res in results]

    def configure_review(
        self,
        server_round: int,
        review_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
        parameters_aggregated: List[Optional[Parameters]],
        metrics_aggregated: List[Dict[str, Scalar]],
    ) -> List[Tuple[ClientProxy, ReviewIns]]:
        # Make instructions and configurations
        config = {}
        if self.on_review_config_fn is not None:
            # Custom review config function provided
            config = self.on_review_config_fn(server_round)
        # Serialize current gloabal model and candidates
        serialized = serialize_global_model_and_candidates(
            parameters.tensors[0],
            [params.tensors[0] for params in parameters_aggregated],
        )
        review_ins = ReviewIns(
            Parameters(tensors=[serialized], tensor_type="bytes"), config
        )

        # Return client/config pairs
        return [(client, review_ins) for client in self.current_round_clients]

    def aggregate_review(
        self,
        server_round: int,
        review_round: int,
        results: List[Tuple[ClientProxy, ReviewRes]],
        failures: List[Union[Tuple[ClientProxy, ReviewRes], BaseException]],
        parameters: Parameters,
        parameters_aggregated: List[Optional[Parameters]],
        metrics_aggregated: List[Dict[str, Scalar]],
    ) -> List[Tuple[Optional[Parameters], Dict[str, Scalar]]]:
        if not results:
            raise AggregateReviewException
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            raise AggregateReviewException
        # Compute gamma
        gamma_numerator = sum(
            [
                parameters_to_ndarrays(review_res.parameters)[0]
                for _, review_res in results
            ]
        )
        gamma_denominator = sum(
            [
                parameters_to_ndarrays(review_res.parameters)[1]
                for _, review_res in results
            ]
        )
        gamma = np.linalg.inv(gamma_denominator) @ gamma_numerator
        gamma = gamma / (len(gamma))
        # Return candidates parameters along with gamma
        return [
            (params, gamma_i) for params, gamma_i in zip(parameters_aggregated, gamma)
        ]

    def aggregate_after_review(
        self,
        server_round: int,
        parameters: Parameters,
        parameters_aggregated: List[Optional[Parameters]],
        metrics_aggregated: List[Dict[str, Scalar]],
    ) -> Optional[Parameters]:
        # Update global model with candidated weighted by computed gamma
        model = deserialize_gradient_boosting_regressor(parameters.tensors[0])
        for params, gamma in zip(parameters_aggregated, metrics_aggregated):
            model.add_estimator(
                deserialize_decision_tree_regressor(params.tensors[0]), gamma
            )
        # Return the new serialized global model
        return Parameters(
            tensors=[serialize_gradient_boosting_regressor(model)], tensor_type="bytes"
        )

    def stop_review(
        self,
        server_round: int,
        review_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
        parameters_aggregated: List[Optional[Parameters]],
        metrics_aggregated: List[Dict[str, Scalar]],
    ) -> bool:
        # Always stop the review process after the first review round
        return True

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        model = FederatedGradientBoostingRegressor()
        return Parameters(
            tensors=[serialize_gradient_boosting_regressor(model)], tensor_type="bytes"
        )

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        eval_res = self.evaluate_fn(server_round, parameters, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics


def main(args):
    _, x_test, _, y_test = load_data(-1, args.num_clients)

    def evaluate_fn(
        server_round: int,
        parameters: Parameters,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model = deserialize_gradient_boosting_regressor(parameters.tensors[0])
        y_pred = model.predict(x_test)
        loss = mean_squared_error(y_test, y_pred)
        return loss, {}

    strategy = FederatedGradientBoostingStrategy(
        fraction_train=args.fraction_train,
        fraction_review=args.fraction_review,
        fraction_evaluate=args.fraction_eval,
        min_train_clients=int(args.num_clients * args.fraction_train),
        min_review_clients=int(args.num_clients * args.fraction_review),
        min_evaluate_clients=int(args.num_clients * args.fraction_eval),
        min_available_clients=args.num_clients,
        evaluate_fn=None if args.fraction_eval > 0 else evaluate_fn,
    )
    server = PeerReviewServer(strategy=strategy)
    return fl.server.start_server(
        server_address=args.server_address,
        server=server,
        config=ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server_address",
        type=str,
        default=os.environ.get("SERVER_ADDRESS", "localhost:8080"),
        required=False,
        help="ip address of the server formatted as: <host:port>",
    )
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=os.environ.get("NUM_ROUNDS", 5),
        required=False,
        help="number of rounds of federated learning to be performed",
    )
    parser.add_argument(
        "--num_clients",
        type=int,
        default=os.environ.get("NUM_CLIENTS", None),
        required=False,
        help="number of clients participating to the training",
    )
    parser.add_argument(
        "--fraction_train",
        type=float,
        default=os.environ.get("FRACTION_TRAIN", 1 / 3),
        required=False,
        help="maximum number of leaves for weak learners trees",
    )
    parser.add_argument(
        "--fraction_review",
        type=float,
        default=os.environ.get("FRACTION_REVIEW", 1 / 3),
        required=False,
        help="maximum number of leaves for weak learners trees",
    )
    parser.add_argument(
        "--fraction_eval",
        type=float,
        default=os.environ.get("FRACTION_EVAL", 1),
        required=False,
        help="maximum number of leaves for weak learners trees",
    )
    parser.add_argument(
        "--max_leaves",
        type=int,
        default=os.environ.get("MAX_LEAVES", 8),
        required=False,
        help="maximum number of leaves for weak learners trees",
    )
    args = parser.parse_args()
    if args.num_clients is None:
        raise ValueError(
            "It is necessary to specify the number of clients participating to the training."
        )
    main(args)
