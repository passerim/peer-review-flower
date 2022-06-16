import concurrent
from logging import DEBUG, ERROR, INFO, WARNING
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from flwr.common import FitRes, Parameters, Scalar
from flwr.common.logger import log
from flwr.common.parameter import weights_to_parameters
from flwr.common.typing import FitIns, ParametersRes
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
from flwr.server.server import Server, fit_clients
from overrides import overrides
from prflwr.peer_review.config import PrConfig
from prflwr.peer_review.strategy.strategy import PeerReviewStrategy
from prflwr.utils.timer import FitTimer


class PeerReviewServer(Server):
    """Implementation of a federated learning server that supports an experimental
    peer review mechanism of model updates based on trained parameters received by
    worker-clients and evaluation of these parameters over multiple review rounds.
    """

    def __init__(
        self,
        client_manager: ClientManager,
        strategy: PeerReviewStrategy,
        max_workers: Optional[int] = None,
        max_review_rounds: int = PrConfig.MAX_REVIEW_ROUNDS,
    ) -> None:
        super().__init__(client_manager, strategy)
        if isinstance(strategy, PeerReviewStrategy):
            self.max_review_rounds = max_review_rounds
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
        for current_round in range(1, num_rounds + 1):
            # Train model on clients and replace previous global model
            parameters_aggregated, metrics_aggregated = self.fit_round(
                current_round, timeout
            )
            # Multiple reviews loop
            for current_review in range(1, self.max_review_rounds + 1):
                if parameters_aggregated is not None:
                    parameters_aggregated, metrics_aggregated = self.review_round(
                        rnd=current_round,
                        review_rnd=current_review,
                        parameters_aggregated=parameters_aggregated,
                        metrics_aggregated=metrics_aggregated,
                        timeout=timeout,
                    )
                if (parameters_aggregated is None) or self.strategy.stop_review(
                    current_round,
                    current_review,
                    self.parameters,
                    self.client_manager,
                    parameters_aggregated,
                    metrics_aggregated,
                ):
                    break
            # Aggregate training results
            self.aggregate_parameters(
                current_round, parameters_aggregated, metrics_aggregated
            )
            # Evaluate model using strategy implementation
            self.evaluate_centralized(current_round, history, timer)
            # Evaluate model on a sample of available clients
            self.evaluate_on_clients(current_round, history, timeout)
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
            res = self.strategy.evaluate(parameters=parameters)
            if res is not None:
                log(
                    INFO,
                    "initial parameters (loss, other metrics): %s, %s",
                    res[0],
                    res[1],
                )
                history.add_loss_centralized(rnd=0, loss=res[0])
                history.add_metrics_centralized(rnd=0, metrics=res[1])
        return parameters

    @staticmethod
    def is_weights_type(weights: List[Any]):
        if isinstance(weights, list):
            if all(map(lambda ndarr: isinstance(ndarr, np.ndarray), weights)):
                return True
        return False

    @staticmethod
    def is_parameters_type(parameters: Any):
        return True if isinstance(parameters, Parameters) else False

    def aggregate_parameters(
        self,
        current_round: int,
        parameters_aggregated: List[Parameters],
        metrics_aggregated: List[Dict[str, Scalar]],
    ) -> bool:
        parameters_prime = self.strategy.aggregate_after_review(
            current_round, parameters_aggregated, metrics_aggregated, self.parameters
        )
        if parameters_prime is None:
            log(
                WARNING,
                """Aggregated parameters are empty!
                Skipping this round of federated learning""",
            )
            return False
        elif self.is_weights_type(parameters_prime):
            parameters_prime = weights_to_parameters(parameters_prime)
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
        self, current_round: int, history: History, timer: FitTimer
    ) -> None:
        res_cen = self.strategy.evaluate(parameters=self.parameters)
        if res_cen:
            loss_cen, metrics_cen = res_cen
            log(
                INFO,
                "fit progress: (\n\tcurrent_round: %s,\n\tcentralized_loss: %s,\n\tmetrics: %s,\n\ttime_elapsed %s\n)",
                current_round,
                loss_cen,
                metrics_cen,
                timer.get_elapsed(),
            )
            history.add_loss_centralized(rnd=current_round, loss=loss_cen)
            history.add_metrics_centralized(rnd=current_round, metrics=metrics_cen)

    def evaluate_on_clients(
        self, current_round: int, history: History, timeout: Optional[float]
    ) -> None:
        res_fed = self.evaluate_round(rnd=current_round, timeout=timeout)
        if res_fed:
            loss_fed, evaluate_metrics_fed, _ = res_fed
            if loss_fed:
                history.add_loss_distributed(rnd=current_round, loss=loss_fed)
                history.add_metrics_distributed(
                    rnd=current_round, metrics=evaluate_metrics_fed
                )

    @staticmethod
    def check_train(
        results: List[Tuple[ClientProxy, FitRes]], failures: List[BaseException]
    ):
        for client, result in results:
            if isinstance(result, FitRes):
                metrics = result.metrics
                if metrics.get(PrConfig.REVIEW_FLAG):
                    results.remove((client, result))
                    failures.append(BaseException())
        return results, failures

    @staticmethod
    def check_review(
        results: List[Tuple[ClientProxy, FitRes]], failures: List[BaseException]
    ):
        for client, result in results:
            if isinstance(result, FitRes):
                metrics = result.metrics
                if not metrics.get(PrConfig.REVIEW_FLAG):
                    results.remove((client, result))
                    failures.append(BaseException())
        return results, failures

    def fit_round(
        self, rnd: int, timeout: Optional[float]
    ) -> Tuple[Optional[List[Parameters]], List[Dict[str, Scalar]]]:
        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_train(
            rnd=rnd, parameters=self.parameters, client_manager=self._client_manager
        )
        if not isinstance(client_instructions, list) or len(client_instructions) < 1:
            log(INFO, "train_round: no clients selected, cancel")
            return None, []
        log(
            DEBUG,
            "train_round: strategy sampled %s clients (out of %s)",
            len(client_instructions),
            self._client_manager.num_available(),
        )
        # Collect training results from all clients participating in this round
        results, failures = fit_clients(
            client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )
        results, failures = self.check_train(results, failures)
        log(
            DEBUG,
            "train_round received %s results and %s failures",
            len(results),
            len(failures),
        )
        # Aggregate training results
        aggregated_result = self.strategy.aggregate_train(rnd, results, failures)
        if not isinstance(aggregated_result, list) or len(aggregated_result) < 1:
            log(WARNING, "Aggregated train result cannot be empty!")
            return None, []
        else:
            parameters_aggregated = [res[0] for res in aggregated_result]
            metrics_aggregated = [res[1] for res in aggregated_result]
        return parameters_aggregated, metrics_aggregated

    @staticmethod
    def add_review_flag_if_missing(
        review_instructions: List[Tuple[ClientProxy, FitIns]]
    ):
        def add_flag(fit_ins: FitIns):
            assert isinstance(fit_ins, FitIns)
            fit_ins.config.setdefault(PrConfig.REVIEW_FLAG, True)
            return fit_ins

        return list(map(lambda ins: (ins[0], add_flag(ins[1])), review_instructions))

    def review_round(
        self,
        rnd: int,
        review_rnd: int,
        parameters_aggregated: List[Parameters],
        metrics_aggregated: List[Dict[str, Scalar]],
        timeout: Optional[float],
    ) -> Tuple[Optional[List[Parameters]], List[Dict[str, Scalar]]]:
        # Get clients and their respective review instructions from strategy
        review_instructions = self.strategy.configure_review(
            rnd=rnd,
            review_rnd=review_rnd,
            parameters=self.parameters,
            client_manager=self._client_manager,
            parameters_aggregated=parameters_aggregated,
            metrics_aggregated=metrics_aggregated,
        )
        if not isinstance(review_instructions, list) or len(review_instructions) < 1:
            log(INFO, "review_round: no clients selected, cancel")
            return None, []
        review_instructions = self.add_review_flag_if_missing(review_instructions)
        log(
            DEBUG,
            "review_round: strategy sampled %s clients (out of %s)",
            len(review_instructions),
            self._client_manager.num_available(),
        )
        # Collect review results from all clients participating in this round.
        results, failures = fit_clients(
            review_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )
        results, failures = self.check_review(results, failures)
        log(
            DEBUG,
            "review_round received %s results and %s failures",
            len(results),
            len(failures),
        )
        # Aggregate review results
        aggregated_result = self.strategy.aggregate_review(
            rnd, review_rnd, results, failures
        )
        if not isinstance(review_instructions, list):
            log(WARNING, "Aggregated review result is invalid!")
            return None, []
        elif len(aggregated_result) >= 1:
            parameters_aggregated = [res[0] for res in aggregated_result]
            metrics_aggregated = [res[1] for res in aggregated_result]
        return parameters_aggregated, metrics_aggregated

    def _get_initial_parameters(self, timeout: Optional[float]) -> Optional[Parameters]:
        """Get initial parameters from one of the available clients."""

        # Server-side parameter initialization
        parameters: Optional[Parameters] = self.strategy.initialize_parameters(
            client_manager=self._client_manager
        )
        if parameters is not None:
            log(INFO, "Using initial parameters provided by strategy")
            return parameters

        # Get initial parameters from one of the clients
        log(INFO, "Requesting initial parameters from one random client")
        random_client = self._client_manager.sample(1)[0]
        parameters_res = get_parameters_from_client(random_client, timeout=timeout)
        log(INFO, "Received initial parameters from one random client")
        return parameters_res.parameters if parameters_res else None


def get_parameters_from_client(
    random_client: ClientProxy,
    timeout: Optional[float],
) -> Optional[ParametersRes]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        submitted_fs = {
            executor.submit(lambda: random_client.get_parameters(timeout=timeout))
        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,  # Handled in the respective communication stack
        )
    future = finished_fs.pop()
    failure = future.exception()
    if failure is not None:
        return None
    else:
        # Success case
        result = future.result()
        return result
