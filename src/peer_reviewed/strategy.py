from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple
from copy import deepcopy
import numpy as np

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
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

from .prstrategy import PeerReviewStrategy
from .prconfig import REVIEW_FLAG


class PeerReviewedFedAvg(FedAvg, PeerReviewStrategy):
    """Peer Reviewed FedAvg strategy implementation."""

    def __init__(
        self,
        fraction_review: float = 0.3,
        fraction_fit: float = 0.1,
        fraction_eval: float = 0.1,
        min_fit_clients: int = 2,
        min_eval_clients: int = 2,
        min_available_clients: int = 2,
        eval_fn: Optional[
            Callable[[Weights], Optional[Tuple[float, Dict[str, Scalar]]]]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
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


    def __repr__(self) -> str:
        rep = f"PeerReviewedFedAvg(accept_failures={self.accept_failures})"
        return rep


    def _aggregate(
        self, 
        weights: List[Tuple[Weights, int]]
    ) -> List[Weights]:
        return aggregate(weights)
 

    def aggregate_and_configure_review(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> List[Tuple[ClientProxy, List[Tuple[Optional[Parameters], Dict[str, Scalar]]]]]:
        """Aggregate train results and configure review."""

        if not results:
            return None
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None

        review_instructions = []
        num_results = len(results)
        idxs = list(range(num_results))
        while len(idxs) > 0:
            if len(idxs) > int(num_results*self.fraction_review):
                curr_idxs = np.random.choice(
                    idxs, 
                    size=int(num_results*self.fraction_review), 
                    replace=False
                )
            else:
                curr_idxs = deepcopy(idxs)
            aggregated_result = self._aggregate(
                [
                    (
                        parameters_to_weights(result.parameters), 
                        result.num_examples
                    )
                    for client, result 
                    in map(results.__getitem__, curr_idxs)
                ]
            )
            review_ins = FitIns(
                weights_to_parameters(aggregated_result), 
                {REVIEW_FLAG: True}
            )
            curr_instructions = [
                (client, review_ins) 
                for client, result 
                in map(results.__getitem__, curr_idxs)
            ]
            review_instructions.extend(curr_instructions)
            for idx in curr_idxs:
                idxs.remove(idx)
        
        return review_instructions
