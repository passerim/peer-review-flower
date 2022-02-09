from logging import DEBUG, INFO, WARNING
from typing import Dict, List, Optional, Tuple

import flwr as fl
from flwr.common import (
    Disconnect,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    Weights,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.server import fit_clients, Server, FitResultsAndFailures

from .prconfig import REVIEW_FLAG
from .prstrategy import PeerReviewStrategy


class PeerReviewServer(Server):

    def __init__(self, client_manager: ClientManager, 
                 strategy: PeerReviewStrategy) -> None:
        super().__init__(client_manager, strategy)

    def fit_round(self, rnd: int) -> Optional[Tuple[Optional[Parameters], 
                                                    Dict[str, Scalar], 
                                                    FitResultsAndFailures]]:

        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_fit(
            rnd=rnd, parameters=self.parameters, client_manager=self._client_manager
        )

        if not client_instructions:
            log(
                INFO, "train_round: no clients selected, cancel"
            )
            return None
        log(
            DEBUG, "train_round: strategy sampled %s clients (out of %s)",
            len(client_instructions), self._client_manager.num_available()
        )

        # Collect `fit` results from all clients participating in this round
        results, failures = fit_clients(
            client_instructions
        )
        
        for client, result in results:
            if isinstance(result, FitRes):
                metrics = result.metrics
                if metrics.get(REVIEW_FLAG):
                    results.remove((client, result))
                    failures.append(BaseException())  

        log(
            DEBUG, "train_round received %s results and %s failures",
            len(results), len(failures)
        )

        # Aggregate training results an get review instructions from strategy
        client_instructions = self.strategy.aggregate_and_configure_review(
            rnd, results, failures
        )

        if not client_instructions:
            log(
                INFO, "review_round: no clients selected, cancel"
            )
            return None
        log(
            DEBUG, "review_round: strategy sampled %s clients (out of %s)",
            len(client_instructions), self._client_manager.num_available()
        )

        # Collect `evaluate` results from all clients participating in this round,
        # these will in practice be the results of the review round.
        results, failures = fit_clients(
            client_instructions
        )

        for client, result in results:
            if isinstance(result, FitRes):
                metrics = result.metrics
                if not metrics.get(REVIEW_FLAG):
                    results.remove((client, result))
                    failures.append(BaseException()) 

        log(
            DEBUG, "review_round received %s results and %s failures",
            len(results), len(failures),
        )

        # Aggregate review results
        aggregated_result = self.strategy.aggregate_fit(rnd, results, failures)

        metrics_aggregated = {}
        if aggregated_result is None:
            log(
                WARNING, "Aggregated result cannot be empty!"
            )
            parameters_aggregated = None
        parameters_aggregated, metrics_aggregated = aggregated_result

        return parameters_aggregated, metrics_aggregated, (results, failures)
