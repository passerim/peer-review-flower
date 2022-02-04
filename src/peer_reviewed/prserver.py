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

from prstrategy import PeerReviewStrategy


FitResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, FitRes]],
    List[BaseException],
]
EvaluateResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, EvaluateRes]],
    List[BaseException],
]
ReconnectResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, Disconnect]],
    List[BaseException],
]


class PeerReviewServer(fl.server.Server):

    def __init__(self, client_manager: ClientManager, 
                 strategy: PeerReviewStrategy) -> None:
        super.__init__(client_manager, strategy)

    def fit_round(self, rnd: int) -> Optional[Tuple[Optional[Parameters], 
                                                    Dict[str, Scalar], 
                                                    FitResultsAndFailures]]:

        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_fit(
            rnd=rnd, parameters=self.parameters, client_manager=self._client_manager
        )

        if not client_instructions:
            log(
                INFO, "fit_round: no clients selected, cancel"
            )
            return None
        log(
            DEBUG, "fit_round: strategy sampled %s clients (out of %s)",
            len(client_instructions), self._client_manager.num_available()
        )

        # Collect `fit` results from all clients participating in this round
        results, failures = self.fit_clients(
            client_instructions,
            max_workers=self.max_workers,
        )

        log(
            DEBUG, "fit_round received %s results and %s failures",
            len(results), len(failures)
        )

        # Aggregate training results
        aggregated_result = self.strategy.aggregate_fit(rnd, results, failures)

        metrics_aggregated = {}
        if aggregated_result is None:
            log(
                WARNING, "Aggregated result cannot be empty!"
            )
            parameters_aggregated = None
        parameters_aggregated, metrics_aggregated = aggregated_result

        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_fit_eval(
            rnd=rnd, parameters=parameters_aggregated, client_instructions=client_instructions
        )

        if not client_instructions:
            log(
                INFO, "fit_round: no clients selected, cancel"
            )
            return None
        log(
            DEBUG, "fit_round: strategy sampled %s clients (out of %s)",
            len(client_instructions), self._client_manager.num_available()
        )

        # Collect `evaluate` results from all clients participating in this round
        results, failures = self.evaluate_clients(
            client_instructions,
            max_workers=self.max_workers,
        )

        log(
            DEBUG, "evaluate_round received %s results and %s failures",
            len(results), len(failures),
        )

        # Aggregate training results
        aggregated_result = self.strategy.aggregate_fit_eval(rnd, results, failures)

        metrics_aggregated = {}
        if aggregated_result is None:
            log(
                WARNING, "Aggregated result cannot be empty!"
            )
            parameters_aggregated = None
        parameters_aggregated, metrics_aggregated = aggregated_result

        return parameters_aggregated, metrics_aggregated, (results, failures)
