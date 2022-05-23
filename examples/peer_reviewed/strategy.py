from copy import deepcopy
from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from flwr.common import (
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    Weights,
    parameters_to_weights,
    weights_to_parameters,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate
from overrides.overrides import overrides

from prflwr.peer_reviewed.prstrategy import PeerReviewStrategy
from prflwr.peer_reviewed.prconfig import PrConfig


class PeerReviewedFedAvg(FedAvg, PeerReviewStrategy):
    """Peer Reviewed FedAvg strategy implementation."""

    def __init__(
        self,
        fraction_fit: float = 0.1,
        fraction_review: float = 0.1,
        fraction_eval: float = 0.1,
        min_fit_clients: int = 2,
        min_review_clients: int = 2,
        min_eval_clients: int = 2,
        min_available_clients: int = 2,
        eval_fn: Optional[
            Callable[[Weights], Optional[Tuple[float, Dict[str, Scalar]]]]
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
        fraction_eval : float, optional
            Fraction of clients used during validation. Defaults to 0.1.
        min_fit_clients : int, optional
            Minimum number of clients used during training. Defaults to 2.
        min_eval_clients : int, optional
            Minimum number of clients used during validation. Defaults to 2.
        min_available_clients : int, optional
            Minimum number of total clients in the system. Defaults to 2.
        eval_fn : Callable[[Weights], Optional[Tuple[float, Dict[str, Scalar]]]]
            Optional function used for validation. Defaults to None.
        on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure training. Defaults to None.
        on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure validation. Defaults to None.
        accept_failures : bool, optional
            Whether or not accept rounds containing failures. Defaults to True.
        initial_parameters : Parameters, optional
            Initial global model parameters.
        """
        super().__init__(
            fraction_fit,
            fraction_eval,
            min_fit_clients,
            min_eval_clients,
            min_available_clients,
            eval_fn,
            on_fit_config_fn,
            on_evaluate_config_fn,
            accept_failures,
            initial_parameters,
        )
        self.fraction_review = fraction_review
        self.min_review_clients = min_review_clients
        self.on_review_config_fn = on_review_config_fn

    def __repr__(self) -> str:
        rep = f"PeerReviewedFedAvg(accept_failures={self.accept_failures})"
        return rep

    @overrides
    def configure_train(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        return super().configure_fit(rnd, parameters, client_manager)

    @overrides
    def aggregate_train(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> List[Tuple[Optional[Parameters], Dict[str, Scalar]]]:
        return super().aggregate_fit(rnd, results, failures)

    @overrides
    def configure_review(
        self,
        rnd: int,
        review_rnd: int,
        parameters: Parameters,
        client_manager: ClientManager,
        parameters_aggregated: List[Optional[Parameters]],
        metrics_aggregated: List[Dict[str, Scalar]],
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of review."""
        if not isinstance(parameters_aggregated, list):
            parameters_aggregated = [parameters_aggregated]
        config = {}

        # Use custom review config function if provided
        if self.on_review_config_fn is not None:
            config = self.on_review_config_fn(rnd)

        # Set review flag
        config.setdefault(PrConfig.REVIEW_FLAG, True)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        if len(parameters_aggregated) > sample_size:
            log(
                WARNING,
                """List of aggregated parameters is longer than
                    the number of clients sampled for the review round.""",
            )
            return None
        review_instructions = []
        num_aggregates = len(parameters_aggregated)
        idxs = list(range(sample_size))
        curr_agg = 0
        while len(idxs) > 0:
            if len(idxs) > int(num_aggregates * self.fraction_review):
                curr_idxs = np.random.choice(
                    idxs, size=int(num_aggregates * self.fraction_review), replace=False
                )
            else:
                curr_idxs = deepcopy(idxs)
            aggregate = parameters_aggregated[curr_agg]
            review_ins = FitIns(aggregate, config)
            curr_instructions = [
                (client, review_ins) for client in map(clients.__getitem__, curr_idxs)
            ]
            review_instructions.extend(curr_instructions)
            for idx in curr_idxs:
                idxs.remove(idx)
        return review_instructions

    @overrides
    def aggregate_review(
        self,
        rnd: int,
        review_rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> List[Tuple[Optional[Parameters], Dict[str, Scalar]]]:

        # Do not aggregate if there are no results or
        # if there are failures and failures are not accepted
        if not results:
            return None
        if not self.accept_failures and failures:
            return None

        # Aggregate review round results
        aggregated_result = aggregate(
            [
                (parameters_to_weights(result.parameters), 1)
                for client, result in results
            ]
        )
        aggregated_result = weights_to_parameters(aggregated_result)
        return [aggregated_result], [{}]

    @overrides
    def aggregate_after_review(
        self,
        rnd: int,
        parameters_aggregated: List[Optional[Parameters]],
        metrics_aggregated: List[Dict[str, Scalar]],
        parameters: Optional[Parameters] = None,
    ) -> Optional[Parameters]:
        aggregated_result = aggregate(
            [
                (parameters_to_weights(parameters), 1)
                for parameters in parameters_aggregated
            ]
        )
        return weights_to_parameters(aggregated_result)

    @overrides
    def stop_review(
        self,
        rnd: int,
        review_rnd: int,
        parameters: Parameters,
        client_manager: ClientManager,
        parameters_aggregated: List[Optional[Parameters]],
        metrics_aggregated: List[Dict[str, Scalar]],
    ) -> bool:
        """Stop condition to decide whether or not to continue with another review round."""
        return True
