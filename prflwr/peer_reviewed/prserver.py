from logging import DEBUG, INFO, WARNING
from typing import Dict, List, Optional, Tuple
from overrides import overrides

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
        if isinstance(strategy, PeerReviewStrategy):
            super().__init__(client_manager, strategy)
        else:
            self.fit_round = super().fit_round

    def _check_train(self, results: List[Tuple[ClientProxy, FitRes]], failures: List[BaseException]):
        for client, result in results:
            if isinstance(result, FitRes):
                metrics = result.metrics
                if metrics.get(REVIEW_FLAG):
                    results.remove((client, result))
                    failures.append(BaseException())
        return results, failures

    def _check_review(self, results: List[Tuple[ClientProxy, FitRes]], failures: List[BaseException]):
        for client, result in results:
            if isinstance(result, FitRes):
                metrics = result.metrics
                if not metrics.get(REVIEW_FLAG):
                    results.remove((client, result))
                    failures.append(BaseException())
        return results, failures

    @overrides
    def fit_round(self, rnd: int) -> Optional[Tuple[Optional[Parameters], 
                                                    Dict[str, Scalar], 
                                                    FitResultsAndFailures]]:

        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_train(
            rnd=rnd, parameters=self.parameters, client_manager=self._client_manager
        )
        if not client_instructions:
            log(INFO, "train_round: no clients selected, cancel")
            return None
        log(
            DEBUG, "train_round: strategy sampled %s clients (out of %s)",
            len(client_instructions), self._client_manager.num_available()
        )

        # Collect training results from all clients participating in this round
        results, failures = fit_clients(
            client_instructions,
            max_workers=self.max_workers,
        )
        results, failures = self._check_train(results, failures)
        log(
            DEBUG, "train_round received %s results and %s failures",
            len(results), len(failures)
        )

        # Aggregate training results
        aggregated_result = self.strategy.aggregate_train(rnd, results, failures)
        metrics_aggregated = {}
        if aggregated_result is None:
            log(
                WARNING, "Aggregated result cannot be empty!"
            )
            parameters_aggregated = None
        else:
            parameters_aggregated, metrics_aggregated = aggregated_result

        # Do-While loop
        while True:

            # Get clients and their respective review instructions from strategy
            review_instructions = self.strategy.configure_review(
                rnd=rnd, 
                parameters=self.parameters, 
                client_manager=self._client_manager, 
                parameters_aggregated=parameters_aggregated, 
                metrics_aggregated=metrics_aggregated
            )
            if not review_instructions:
                log(INFO, "review_round: no clients selected, cancel"
                )
                return None
            log(
                DEBUG, "review_round: strategy sampled %s clients (out of %s)",
                len(review_instructions), self._client_manager.num_available()
            )

            # Collect review results from all clients participating in this round.
            results, failures = fit_clients(
                client_instructions,
                max_workers=self.max_workers,
            )
            results, failures = self._check_review(results, failures) 
            log(
                DEBUG, "review_round received %s results and %s failures",
                len(results), len(failures),
            )

            # Aggregate review results
            aggregated_result = self.strategy.aggregate_review(rnd, results, failures)
            metrics_aggregated = {}
            if aggregated_result is None:
                log(
                    WARNING, "Aggregated result cannot be empty!"
                )
                parameters_aggregated = None
            else:
                parameters_aggregated, metrics_aggregated = aggregated_result

            # Stop condition, while...
            if self.strategy.stop_review(
                rnd=rnd, parameters=self.parameters, client_manager=self.client_manager,
                parameters_aggregated=parameters_aggregated, metrics_aggregated=metrics_aggregated
            ): 
                break

        # Aggregate training results
        parameters_aggregated = self.strategy.aggregate_after_review(
            rnd, parameters_aggregated, self.parameters
        )
        if parameters_aggregated is None:
            log(
                WARNING, """Aggregated parameters are empty! 
                Skipping this round of federated learning"""
            )
        return parameters_aggregated, None, None
