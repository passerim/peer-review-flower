from abc import abstractmethod
from typing import Dict, List, Optional, Tuple

from flwr.common import (
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    Weights,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy


class PeerReviewStrategy(Strategy):


    @abstractmethod
    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate peer review results.
        
        Parameters
        ----------
        rnd : int
            The current round of federated learning.
        results : List[Tuple[ClientProxy, FitRes]]
            Successful reviews from the previously selected and configured
            clients. Each pair of `(ClientProxy, FitRes)` constitutes a
            successful review from one of the previously selected clients. Not
            that not all previously selected clients are necessarily included in
            this list: a client might drop out and not submit a result. For each
            client that did not submit an update, there should be an `Exception`
            in `failures`.
        failures : List[BaseException]
            Exceptions that occurred while the server was waiting for client
            updates.
        Returns
        -------
        parameters: Parameters (optional)
            If parameters are returned, then the server will treat these as the
            new global model parameters (i.e., it will replace the previous
            parameters with the ones returned from this method). If `None` is
            returned (e.g., because there were only failures and no viable
            results) then the server will no update the previous model
            parameters, the updates received in this round are discarded, and
            the global model parameters remain the same.
        """


    @abstractmethod
    def aggregate_and_configure_review(
        self, 
        rnd: int, 
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Aggregate train results and configure the next round of peer review.

        Parameters
        ----------
        rnd : int
            The current round of federated learning.
        results : List[Tuple[ClientProxy, FitRes]]
            Successful updates from the previously selected and configured
            clients. Each pair of `(ClientProxy, FitRes)` constitutes a
            successful trained model update from one of the previously selected clients. 
            Note that not all previously selected clients are necessarily included in
            this list: a client might drop out and not submit a result. For each
            client that did not submit an update, there should be an `Exception`
            in `failures`.
        failures : List[BaseException]
            Exceptions that occurred while the server was waiting for client
            updates.

        Returns
        -------
        review_instructions: List[Tuple[ClientProxy, FitIns]]
            A list of tuples. Each tuple in the list identifies a `ClientProxy` and the
            `FitIns` for this particular `ClientProxy`. If a particular `ClientProxy`
            is not included in this list, it means that this `ClientProxy`
            will not participate in the current round of review.
            The `FitIns` must include a flag to tell the clients this is a review round.
        """
