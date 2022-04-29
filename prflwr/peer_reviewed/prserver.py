from logging import DEBUG, INFO, WARNING
from typing import Dict, List, Optional, Tuple

from flwr.common import FitRes, Parameters, Scalar
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
from flwr.server.server import FitResultsAndFailures, Server, fit_clients
from overrides import overrides

from ..utils.timer import FitTimer
from .prconfig import MAX_REVIEW_ROUNDS, REVIEW_FLAG
from .prstrategy import PeerReviewStrategy


class PeerReviewServer(Server):
    def __init__(
        self,
        client_manager: ClientManager,
        strategy: PeerReviewStrategy,
        max_workers: Optional[int] = None,
        max_review_rounds: int = MAX_REVIEW_ROUNDS,
    ) -> None:
        super().__init__(client_manager, strategy)
        if isinstance(strategy, PeerReviewStrategy):
            self.max_review_rounds = max_review_rounds
        else:
            self.fit_round = super().fit_round
            self.fit = super().fit
        self.set_max_workers(max_workers)

    def fit(self, num_rounds: int) -> History:
        """Run federated learning with peer review for a number of rounds."""
        history = History()
        # Initialize parameters
        self.init_parameters(history)
        # Run federated learning for num_rounds rounds
        log(INFO, "FL starting")
        timer = FitTimer()
        for current_round in range(1, num_rounds + 1):
            # Train model on clients and replace previous global model
            parameters_aggregated, metrics_aggregated, _ = self.fit_round(current_round)
            # Multiple reviews loop
            for current_review in range(self.max_review_rounds):
                parameters_aggregated, metrics_aggregated, _ = self.review_round(
                    current_round,
                    current_review,
                    parameters_aggregated,
                    metrics_aggregated,
                )
                if self.strategy.stop_review(
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
            self.evaluate_on_clients(current_round, history)
        # Bookkeeping
        log(INFO, "FL finished in %s", timer.get_elapsed())
        return history

    def init_parameters(self, history: History) -> None:
        log(INFO, "Initializing global parameters")
        self.parameters = self._get_initial_parameters()
        log(INFO, "Evaluating initial parameters")
        res = self.strategy.evaluate(parameters=self.parameters)
        if res is not None:
            log(
                INFO,
                "initial parameters (loss, other metrics): %s, %s",
                res[0],
                res[1],
            )
            history.add_loss_centralized(rnd=0, loss=res[0])
            history.add_metrics_centralized(rnd=0, metrics=res[1])

    def aggregate_parameters(
        self,
        current_round: int,
        parameters_aggregated: List[Parameters],
        metrics_aggregated: List[Dict[str, Scalar]],
    ) -> None:
        parameters_aggregated = self.strategy.aggregate_after_review(
            current_round, parameters_aggregated, metrics_aggregated, self.parameters
        )
        if parameters_aggregated is None:
            log(
                WARNING,
                """Aggregated parameters are empty!
                Skipping this round of federated learning""",
            )
        else:
            self.parameters = parameters_aggregated

    def evaluate_centralized(self, current_round: int, history: History, timer: FitTimer) -> None:
        res_cen = self.strategy.evaluate(parameters=self.parameters)
        if res_cen is not None:
            loss_cen, metrics_cen = res_cen
            log(
                INFO,
                "fit progress: (%s, %s, %s, %s)",
                current_round,
                loss_cen,
                metrics_cen,
                timer.get_elapsed(),
            )
            history.add_loss_centralized(rnd=current_round, loss=loss_cen)
            history.add_metrics_centralized(rnd=current_round, metrics=metrics_cen)

    def evaluate_on_clients(self, current_round: int, history: History) -> None:
        res_fed = self.evaluate_round(rnd=current_round)
        if res_fed:
            loss_fed, evaluate_metrics_fed, _ = res_fed
            if loss_fed:
                history.add_loss_distributed(rnd=current_round, loss=loss_fed)
                history.add_metrics_distributed(
                    rnd=current_round, metrics=evaluate_metrics_fed
                )

    def _check_train(
        self, results: List[Tuple[ClientProxy, FitRes]], failures: List[BaseException]
    ):
        for client, result in results:
            if isinstance(result, FitRes):
                metrics = result.metrics
                if metrics.get(REVIEW_FLAG):
                    results.remove((client, result))
                    failures.append(BaseException())
        return results, failures

    def _check_review(
        self, results: List[Tuple[ClientProxy, FitRes]], failures: List[BaseException]
    ):
        for client, result in results:
            if isinstance(result, FitRes):
                metrics = result.metrics
                if not metrics.get(REVIEW_FLAG):
                    results.remove((client, result))
                    failures.append(BaseException())
        return results, failures

    @overrides
    def fit_round(
        self, rnd: int
    ) -> Tuple[
        List[Parameters], List[Dict[str, Scalar]], FitResultsAndFailures
    ]:
        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_train(
            rnd=rnd, parameters=self.parameters, client_manager=self._client_manager
        )
        if not client_instructions:
            log(INFO, "train_round: no clients selected, cancel")
            # TODO Recover!
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
        )
        results, failures = self._check_train(results, failures)
        log(
            DEBUG,
            "train_round received %s results and %s failures",
            len(results),
            len(failures),
        )

        # Aggregate training results
        aggregated_result = self.strategy.aggregate_train(rnd, results, failures)
        metrics_aggregated = {}
        if aggregated_result is None:
            log(WARNING, "Aggregated result cannot be empty!")
            # TODO Recover!
        else:
            parameters_aggregated, metrics_aggregated = aggregated_result

        return parameters_aggregated, metrics_aggregated, (results, failures)

    def review_round(
        self,
        rnd: int,
        review_rnd: int,
        parameters_aggregated: List[Parameters],
        metrics_aggregated: List[Dict[str, Scalar]],
    ) -> Tuple[
        List[Parameters], List[Dict[str, Scalar]], FitResultsAndFailures
    ]:
        # Get clients and their respective review instructions from strategy
        review_instructions = self.strategy.configure_review(
            rnd=rnd,
            review_rnd=review_rnd,
            parameters=self.parameters,
            client_manager=self._client_manager,
            parameters_aggregated=parameters_aggregated,
            metrics_aggregated=metrics_aggregated,
        )
        if not review_instructions:
            log(INFO, "review_round: no clients selected, cancel")
            # TODO Recover!
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
        )
        results, failures = self._check_review(results, failures)
        log(
            DEBUG,
            "review_round received %s results and %s failures",
            len(results),
            len(failures),
        )

        # Aggregate review results
        aggregated_result = self.strategy.aggregate_review(rnd, results, failures)
        if aggregated_result is None:
            log(WARNING, "Aggregated result cannot be empty!")
            # TODO Recover!
        else:
            parameters_aggregated, metrics_aggregated = aggregated_result

        return parameters_aggregated, metrics_aggregated, (results, failures)
