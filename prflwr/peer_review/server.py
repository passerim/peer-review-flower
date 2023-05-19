from concurrent import futures
from logging import DEBUG, ERROR, INFO, WARNING
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
from flwr.common import (
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
)
from flwr.common.logger import log
from flwr.server import ClientManager, History, Server, SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.server import fit_clients
from overrides import overrides

from prflwr.peer_review.config import PrConfig
from prflwr.peer_review.strategy.fedavg import PeerReviewedFedAvg
from prflwr.peer_review.strategy.strategy import PeerReviewStrategy
from prflwr.peer_review.typing import ReviewIns, TrainIns
from prflwr.utils.timer import FitTimer


class PeerReviewServer(Server):
    """Implementation of a federated learning server that supports an
    experimental peer review mechanism of model updates based on trained
    parameters received by clients and evaluation of these parameters over
    multiple review rounds."""

    def __init__(
        self,
        client_manager: ClientManager = None,
        strategy: PeerReviewStrategy = None,
        max_workers: Optional[int] = None,
        max_review_rounds: int = PrConfig.MAX_REVIEW_ROUNDS,
    ) -> None:
        super().__init__(
            client_manager=client_manager
            if client_manager is not None
            else SimpleClientManager(),
            strategy=strategy if strategy is not None else PeerReviewedFedAvg(),
        )
        if isinstance(strategy, PeerReviewStrategy):
            self.max_review_rounds = max_review_rounds
            self.strategy: PeerReviewStrategy = strategy
        else:
            self.fit_round = super().fit_round
            self.fit = super().fit
        self.set_max_workers(max_workers)

    @overrides
    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        """Run federated learning with peer review for a number of rounds."""
        history = History()

        # Initialize parameters
        self.parameters = self.init_parameters(history, timeout)
        if self.parameters is None:
            log(
                WARNING,
                "Parameters initialization unsuccessful, terminating federated learning!",
            )
            return history

        # Run federated learning for num_rounds rounds
        log(INFO, "FL starting")
        timer = FitTimer().start()
        for server_round in range(1, num_rounds + 1):
            # Train model on clients and replace previous global model
            parameters_aggregated, metrics_aggregated = self.train_round(
                server_round, timeout
            )

            # Multiple reviews loop
            for review_round in range(1, self.max_review_rounds + 1):
                if parameters_aggregated is not None:
                    parameters_aggregated, metrics_aggregated = self.review_round(
                        server_round,
                        review_round,
                        parameters_aggregated,
                        metrics_aggregated,
                        timeout,
                    )

                # Check with strategy whether to stop
                if (parameters_aggregated is None) or self.strategy.stop_review(
                    server_round,
                    review_round,
                    self.parameters,
                    self._client_manager,
                    parameters_aggregated,
                    metrics_aggregated,
                ):
                    break

            # Aggregate training results
            if parameters_aggregated is None:
                log(
                    WARNING,
                    """Aggregated parameters are empty!
                    Skipping this round of federated learning""",
                )
                continue
            self.aggregate_parameters(
                server_round, parameters_aggregated, metrics_aggregated
            )

            # Evaluate model using strategy implementation
            self.evaluate_centralized(server_round, history, timer)

            # Evaluate model on a sample of available clients
            self.evaluate_on_clients(server_round, history, timer, timeout)

        # Bookkeeping
        log(INFO, "FL finished in %s", timer.get_elapsed())
        return history

    def init_parameters(
        self, history: History, timeout: Optional[float]
    ) -> Optional[Parameters]:
        log(INFO, "Initializing global parameters")
        parameters = self._get_initial_parameters(timeout=timeout)
        if parameters:
            log(INFO, "Evaluating initial parameters")
            res = self.strategy.evaluate(server_round=0, parameters=parameters)
            if res is not None:
                log(
                    INFO,
                    "initial parameters (loss, other metrics): %s, %s",
                    res[0],
                    res[1],
                )
                history.add_loss_centralized(server_round=0, loss=res[0])
                history.add_metrics_centralized(server_round=0, metrics=res[1])
        return parameters

    @staticmethod
    def is_weights_type(weights: Any):
        if isinstance(weights, list):
            if all(map(lambda ndarray: isinstance(ndarray, np.ndarray), weights)):
                return True
        return False

    @staticmethod
    def is_parameters_type(parameters: Any):
        return True if isinstance(parameters, Parameters) else False

    def aggregate_parameters(
        self,
        server_round: int,
        parameters_aggregated: List[Parameters],
        metrics_aggregated: List[Dict[str, Scalar]],
    ) -> bool:
        parameters_prime = self.strategy.aggregate_after_review(
            server_round,
            self.parameters,
            parameters_aggregated,
            metrics_aggregated,
        )
        if parameters_prime is None:
            log(
                WARNING,
                """Aggregated parameters are empty!
                Skipping this round of federated learning""",
            )
            return False
        elif self.is_weights_type(parameters_prime):
            parameters_prime = ndarrays_to_parameters(cast(NDArrays, parameters_prime))
        elif not self.is_parameters_type(parameters_prime):
            log(
                ERROR,
                """Aggregated parameters type is incorrect!
                Skipping this round of federated learning""",
            )
            return False
        self.parameters = parameters_prime
        return True

    def evaluate_centralized(
        self, server_round: int, history: History, timer: FitTimer
    ) -> None:
        res_cen = self.strategy.evaluate(server_round, self.parameters)
        if res_cen:
            loss_cen, metrics_cen = res_cen
            log(
                INFO,
                "fit progress: (\n\tserver_round: %s,\n\tcentralized_loss: %s,\n\tmetrics: %s,\n\ttime_elapsed %s\n)",
                server_round,
                loss_cen,
                metrics_cen,
                timer.get_elapsed(),
            )
            history.add_loss_centralized(server_round, loss=loss_cen)
            history.add_metrics_centralized(server_round, metrics=metrics_cen)

    def evaluate_on_clients(
        self,
        server_round: int,
        history: History,
        timer: FitTimer,
        timeout: Optional[float],
    ) -> None:
        res_fed = self.evaluate_round(server_round, timeout)
        if res_fed:
            loss_fed, evaluate_metrics_fed, _ = res_fed
            if loss_fed is None:
                return
            log(
                INFO,
                "fit progress: (\n\tserver_round: %s,\n\tdistributed_loss: %s,\n\tmetrics: %s,\n\ttime_elapsed %s\n)",
                server_round,
                loss_fed,
                evaluate_metrics_fed,
                timer.get_elapsed(),
            )
            history.add_loss_distributed(server_round, loss=loss_fed)
            history.add_metrics_distributed(server_round, metrics=evaluate_metrics_fed)

    @staticmethod
    def _check_train(
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ):
        for client, result in results:
            if isinstance(result, FitRes):
                metrics = result.metrics
                if metrics.get(PrConfig.REVIEW_FLAG):
                    results.remove((client, result))
                    failures.append((client, result))
        return results, failures

    @staticmethod
    def _make_train_instructions(
        client_instructions: List[Tuple[ClientProxy, TrainIns]]
    ) -> List[Tuple[ClientProxy, FitIns]]:
        def add_flag(fit_ins: TrainIns):
            assert isinstance(fit_ins, (FitIns, TrainIns))
            fit_ins.config.setdefault(PrConfig.REVIEW_FLAG, False)
            return fit_ins

        return list(
            map(
                lambda ins: (ins[0], cast(FitIns, add_flag(ins[1]))),
                client_instructions,
            )
        )

    def train_round(
        self, server_round: int, timeout: Optional[float]
    ) -> Tuple[Optional[List[Parameters]], List[Dict[str, Scalar]]]:
        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_train(
            server_round, self.parameters, self._client_manager
        )
        if not isinstance(client_instructions, list) or len(client_instructions) < 1:
            log(INFO, "train_round %s: no clients selected, cancel", server_round)
            return None, []
        client_instructions = self._make_train_instructions(client_instructions)
        log(
            DEBUG,
            "train_round %s: strategy sampled %s clients (out of %s)",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect training results from all clients participating in this round
        results, failures = fit_clients(
            client_instructions,
            self.max_workers,
            timeout,
        )
        results, failures = self._check_train(results, failures)
        log(
            DEBUG,
            "train_round %s received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )

        # Aggregate training results
        aggregated_result = self.strategy.aggregate_train(
            server_round,
            results,
            failures,
            self.parameters,
        )
        if not isinstance(aggregated_result, list):
            log(WARNING, "Aggregated train result cannot be empty!")
            return None, []
        elif len(aggregated_result) > 0:
            parameters_aggregated = [res[0] for res in aggregated_result]
            metrics_aggregated = [res[1] for res in aggregated_result]
        else:
            log(WARNING, "Aggregated train result is empty!")
            parameters_aggregated = []
            metrics_aggregated = []
        return parameters_aggregated, metrics_aggregated

    @staticmethod
    def _make_review_instructions(
        client_instructions: List[Tuple[ClientProxy, ReviewIns]]
    ) -> List[Tuple[ClientProxy, FitIns]]:
        def add_flag(fit_ins: FitIns):
            assert isinstance(fit_ins, (FitIns, ReviewIns))
            fit_ins.config.setdefault(PrConfig.REVIEW_FLAG, True)
            return fit_ins

        return list(
            map(
                lambda ins: (ins[0], cast(FitIns, add_flag(ins[1]))),
                client_instructions,
            )
        )

    @staticmethod
    def _check_review(
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ):
        for client, result in results:
            if isinstance(result, FitRes):
                metrics = result.metrics
                if not metrics.get(PrConfig.REVIEW_FLAG):
                    results.remove((client, result))
                    failures.append((client, result))
        return results, failures

    def review_round(
        self,
        server_round: int,
        review_round: int,
        parameters_aggregated: List[Parameters],
        metrics_aggregated: List[Dict[str, Scalar]],
        timeout: Optional[float],
    ) -> Tuple[Optional[List[Parameters]], List[Dict[str, Scalar]]]:
        # Get clients and their respective review instructions from strategy
        review_instructions = self.strategy.configure_review(
            server_round,
            review_round,
            self.parameters,
            self._client_manager,
            parameters_aggregated,
            metrics_aggregated,
        )
        if not isinstance(review_instructions, list) or len(review_instructions) < 1:
            log(INFO, "review_round %s: no clients selected, cancel", review_round)
            return None, []
        review_instructions = self._make_review_instructions(review_instructions)
        log(
            DEBUG,
            "review_round %s: strategy sampled %s clients (out of %s)",
            review_round,
            len(review_instructions),
            self._client_manager.num_available(),
        )

        # Collect review results from all clients participating in this round.
        results, failures = fit_clients(
            review_instructions,
            self.max_workers,
            timeout,
        )
        results, failures = self._check_review(results, failures)
        log(
            DEBUG,
            "review_round %s received %s results and %s failures",
            review_round,
            len(results),
            len(failures),
        )

        # Aggregate review results
        aggregated_result = self.strategy.aggregate_review(
            server_round,
            review_round,
            results,
            failures,
            self.parameters,
            parameters_aggregated,
            metrics_aggregated,
        )
        if not isinstance(review_instructions, list):
            log(WARNING, "Aggregated review result is invalid!")
            return None, []
        elif len(aggregated_result) > 0:
            parameters_aggregated = [res[0] for res in aggregated_result]
            metrics_aggregated = [res[1] for res in aggregated_result]
        else:
            log(WARNING, "Aggregated review result is empty!")
            parameters_aggregated = []
            metrics_aggregated = []
        return parameters_aggregated, metrics_aggregated

    def _get_initial_parameters(self, timeout: Optional[float]) -> Optional[Parameters]:
        """Get initial parameters from one of the available clients."""

        # Server-side parameter initialization
        parameters: Optional[Parameters] = self.strategy.initialize_parameters(
            self._client_manager
        )
        if parameters is not None:
            log(INFO, "Using initial parameters provided by strategy")
            return parameters

        # Get initial parameters from one of the clients
        log(INFO, "Requesting initial parameters from one random client")
        random_client = self._client_manager.sample(1)[0]
        ins = GetParametersIns(config={})
        parameters_res = get_parameters_from_client(random_client, ins, timeout)
        log(INFO, "Received initial parameters from one random client")
        return parameters_res.parameters if parameters_res else None


def get_parameters_from_client(
    random_client: ClientProxy,
    ins: GetParametersIns,
    timeout: Optional[float],
) -> Optional[GetParametersRes]:
    with futures.ThreadPoolExecutor(max_workers=1) as executor:
        submitted_fs = {
            executor.submit(
                lambda: random_client.get_parameters(ins=ins, timeout=timeout)
            )
        }
        finished_fs, _ = futures.wait(
            fs=submitted_fs,
            timeout=None,  # Handled in the respective communication stack
        )
    future = finished_fs.pop()
    return future.result() if not future.exception() else None
