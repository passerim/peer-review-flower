from typing import Callable, Dict, List, Optional, Tuple, Union

from flwr.common import (
    FitIns,
    FitRes,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate
from overrides.overrides import overrides
from prflwr.peer_review.strategy.strategy import (
    AggregateReviewException,
    ConfigureReviewException,
    PeerReviewStrategy,
)


class PeerReviewedFedAvg(FedAvg, PeerReviewStrategy):
    """Peer Reviewed FedAvg strategy implementation,
    a simple implementation of a strategy which is equivalent
    to Federated Averaging and should be used mainly for testing."""

    def __init__(
        self,
        fraction_fit: float = 0.1,
        fraction_review: float = 0.1,
        fraction_evaluate: float = 0.1,
        min_fit_clients: int = 2,
        min_review_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_review_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
    ) -> None:
        """Peer Reviewed Federated Averaging strategy.

        Parameters
        ----------
        fraction_fit : float, optional
            Fraction of clients used during training. Defaults to 0.1.
        fraction_evaluate : float, optional
            Fraction of clients used during validation. Defaults to 0.1.
        min_fit_clients : int, optional
            Minimum number of clients used during training. Defaults to 2.
        min_evaluate_clients : int, optional
            Minimum number of clients used during validation. Defaults to 2.
        min_available_clients : int, optional
            Minimum number of total clients in the system. Defaults to 2.
        evaluate_fn : Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]]
            ]
        ]
            Optional function used for validation. Defaults to None.
        on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure training. Defaults to None.
        on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure validation. Defaults to None.
        accept_failures : bool, optional
            Whether accept rounds containing failures. Defaults to True.
        initial_parameters : Parameters, optional
            Initial global model parameters.
        """
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
        )
        self.fraction_review = fraction_review
        self.min_review_clients = min_review_clients
        self.on_review_config_fn = on_review_config_fn

    def __repr__(self) -> str:
        rep = f"PeerReviewedFedAvg(accept_failures={self.accept_failures})"
        return rep

    @overrides
    def configure_train(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        return super().configure_fit(server_round, parameters, client_manager)

    @overrides
    def aggregate_train(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
        parameters: Optional[Parameters] = None,
    ) -> List[Tuple[Optional[Parameters], Dict[str, Scalar]]]:
        return [super().aggregate_fit(server_round, results, failures)]

    @overrides
    def configure_review(
        self,
        server_round: int,
        review_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
        parameters_aggregated: List[Optional[Parameters]],
        metrics_aggregated: List[Dict[str, Scalar]],
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of review."""
        if len(parameters_aggregated) != 1:
            raise ConfigureReviewException
        # Use custom review config function if provided
        config = {}
        if self.on_review_config_fn is not None:
            config = self.on_review_config_fn(server_round)
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        # Make review instructions
        review_parameters = parameters_aggregated.pop(0)
        if review_parameters:
            review_ins = FitIns(review_parameters, config)
            return [(client, review_ins) for client in clients]
        else:
            raise ConfigureReviewException

    @overrides
    def aggregate_review(
        self,
        server_round: int,
        review_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
        parameters: Parameters,
        parameters_aggregated: List[Optional[Parameters]],
        metrics_aggregated: List[Dict[str, Scalar]],
    ) -> List[Tuple[Optional[Parameters], Dict[str, Scalar]]]:
        # Do not aggregate if there are no results or
        # if there are failures and failures are not accepted
        if not results:
            raise AggregateReviewException
        if not self.accept_failures and failures:
            raise AggregateReviewException
        # Aggregate review round results
        aggregated_result = aggregate(
            [
                (parameters_to_ndarrays(result.parameters), result.num_examples)
                for _, result in results
            ]
        )
        return [(ndarrays_to_parameters(aggregated_result), {})]

    @overrides
    def aggregate_after_review(
        self,
        server_round: int,
        parameters: Parameters,
        parameters_aggregated: List[Optional[Parameters]],
        metrics_aggregated: List[Dict[str, Scalar]],
    ) -> Optional[Parameters]:
        return parameters_aggregated.pop(0)

    @overrides
    def stop_review(
        self,
        server_round: int,
        review_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
        parameters_aggregated: List[Optional[Parameters]],
        metrics_aggregated: List[Dict[str, Scalar]],
    ) -> bool:
        """Stop condition to decide whether to continue with another review round."""
        return True
