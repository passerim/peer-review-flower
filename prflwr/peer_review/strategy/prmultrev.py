from abc import abstractmethod
from typing import Dict, List, Optional, Tuple, Union

from flwr.common import FitIns, FitRes, Parameters, Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy


class MultipleReviewStrategy(Strategy):
    """Interface for multiple reviews strategy implementations."""

    @abstractmethod
    def configure_train(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of federated training of the global model.

        Parameters
        ----------
        server_round : int
            The current round of federated learning.
        parameters : Parameters
            The current (global) model parameters.
        client_manager : ClientManager
            The client manager which holds all currently connected clients.

        Returns
        -------
        client_instructions : List[Tuple[ClientProxy, FitIns]]
            A list of tuples. Each tuple in the list identifies a `ClientProxy` and the
            `FitIns` for this particular `ClientProxy`. If a particular `ClientProxy`
            is not included in this list, it means that this `ClientProxy` will not
            participate in the next round of federated learning.
        """

    @abstractmethod
    def aggregate_train(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
        parameters: Optional[Parameters] = None,
    ) -> List[Tuple[Optional[Parameters], Dict[str, Scalar]]]:
        """Aggregate training results of the current round of federated learning.

        Parameters
        ----------
        server_round : int
            The current round of federated learning.
        results : List[Tuple[ClientProxy, FitRes]]
            Successful model updates from the previously selected and configured
            clients. Each pair of `(ClientProxy, FitRes)` constitutes a successful
            model update from one of the previously selected clients. Note that not
            all previously selected clients are necessarily included in this list:
            a client might drop out and not submit a result. For each client that
            did not submit an update, there should be an `Exception` in `failures`.
        failures : List[Union[Tuple[ClientProxy, FitRes], BaseException]]
            Exceptions that occurred while the server was waiting for client
            updates.
        parameters : Optional[Parameters]
            The current (global) model parameters.

        Returns
        -------
        aggregated_result : List[Tuple[Optional[Parameters], Dict[str, Scalar]]]
            If a list of parameters and metrics are returned, then the server will
            use these parameters and metrics as the input for the next review step.
            If `None` is returned (e.g., because there were only failures and no viable
            results) then the server will not update the previous model parameters,
            the updates received in this round are discarded, and the global model
            parameters remain the same.
        """

    @abstractmethod
    def configure_review(
        self,
        server_round: int,
        review_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
        parameters_aggregated: List[Optional[Parameters]],
        metrics_aggregated: List[Dict[str, Scalar]],
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of peer review in the current round of federated learning.

        Parameters
        ----------
        server_round : int
            The current round of federated learning.
        review_round : int
            The current review round.
        parameters : Parameters
            The current (global) model parameters.
        client_manager : ClientManager
            The client manager which holds all currently connected clients.
        parameters_aggregated : List[Optional[Parameters]]
            A list of `Parameters` from the previous round of train or review.
        metrics_aggregated : List[Dict[str, Scalar]]
            A list of `Dict` with metrics from the previous round of train
            or review, corresponding to each `Parameters` in the list
            `parameters_aggregated`.

        Returns
        -------
        review_instructions : List[Tuple[ClientProxy, FitIns]]
            A list of tuples. Each tuple in the list identifies a `ClientProxy` and the
            `FitIns` for this particular `ClientProxy`. If a particular `ClientProxy`
            is not included in this list, it means that this `ClientProxy` will not
            participate in the next round of review. The `FitIns` must include a flag
            to tell the clients this will be a review round.
        """

    @abstractmethod
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
        """Aggregate review results.

        Parameters
        ----------
        server_round : int
            The current round of federated learning.
        review_round : int
            The current review round.
        results : List[Tuple[ClientProxy, FitRes]]
            Successful reviews from the previously selected and configured clients.
            Each pair of `(ClientProxy, FitRes)` constitutes a successful review
            from one of the previously selected clients. Note that not all previously
            selected clients are necessarily included in this list: a client might
            drop out and not submit a result. For each client that did not submit
            an update, there should be an `Exception` in `failures`.
        failures : List[Union[Tuple[ClientProxy, FitRes], BaseException]]
            Exceptions that occurred while the server was waiting for client
            updates.
        parameters : Parameters
            The current (global) model parameters.
        parameters_aggregated : List[Optional[Parameters]]
            A list of `Parameters` from the previous round of train or review.
        metrics_aggregated : List[Dict[str, Scalar]]
            A list of `Dict` with metrics from the previous round of train
            or review, corresponding to each `Parameters` in the list
            `parameters_aggregated`.

        Returns
        -------
        aggregated_result : List[Tuple[Optional[Parameters], Dict[str, Scalar]]]
            If parameters are returned, then the server will treat these as the
            new global model candidate parameters, and they will be used in the next
            round of reviewing or as inputs of the post-reviewing aggregation function.
            If `None` is returned (e.g., because there were only failures and no viable
            results) then the server will not update the previous model parameters,
            the updates received in this round are discarded, and the global model
            parameters remain the same.
        """

    @abstractmethod
    def aggregate_after_review(
        self,
        server_round: int,
        parameters: Parameters,
        parameters_aggregated: List[Optional[Parameters]],
        metrics_aggregated: List[Dict[str, Scalar]],
    ) -> Optional[Parameters]:
        """Aggregate results of the last round of review.

        Parameters
        ----------
        server_round : int
            The current round of federated learning.
        parameters : Optional[Parameters]
            The current (global) model parameters.
        parameters_aggregated: List[Optional[Parameters]]
            List of model parameters from successful review results at the end
            of the previous review rounds.
        metrics_aggregated: List[Dict[str, Scalar]],
            List of metrics from successful review results at the end
            of the previous review rounds.

        Returns
        -------
        parameters_prime : Optional[Parameters]
            If parameters are returned, then the server will treat these as the
            new global model parameters (i.e., it will replace the previous
            parameters with the ones returned from this method). If `None` is
            returned (e.g., because there were only failures and no viable
            results) then the server will not update the previous model
            parameters, the updates received in this round are discarded, and
            the global model parameters remain the same.
        """

    @abstractmethod
    def stop_review(
        self,
        server_round: int,
        review_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
        parameters_aggregated: List[Optional[Parameters]],
        metrics_aggregated: List[Dict[str, Scalar]],
    ) -> bool:
        """Stop condition to decide whether to continue or not with another review round.

        Parameters
        ----------
        server_round : int
            The current round of federated learning.
        review_round : int
            The current review round.
        parameters : Parameters
            The current (global) model parameters.
        client_manager : ClientManager
            The client manager which holds all currently connected clients.
        parameters_aggregated : List[Optional[Parameters]]
            Current list of aggregates, candidates for the new (global) model parameters.
        metrics_aggregated : List[Dict[str, Scalar]]
            Metrics associated with the current aggregates.

        Returns
        -------
        stop : bool
            Whether the review process should terminate at the current round or not.
        """
