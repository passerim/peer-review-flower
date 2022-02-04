from typing import Dict, List, Optional, Tuple

from flwr.common import (
    FitIns,
    FitRes,
    Parameters,
    Scalar,
)
from flwr.server.client_manager import ClientManager
from flwr.server.strategy import Strategy
from flwr.server.client_proxy import ClientProxy


class PeerReviewStrategy(Strategy):

    def configure_fit_eval(self, 
                           rnd: int, 
                           parameters: Parameters, 
                           client_manager: ClientManager
                           ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of peer reviewing.
        Parameters
        ----------
        rnd : int
            The current round of federated learning.
        parameters : Parameters
            The current (global) model parameters.
        client_manager : ClientManager
            The client manager which holds all currently connected clients.
        Returns
        -------
        A list of tuples. Each tuple in the list identifies a `ClientProxy` and the
        `FitIns` for this particular `ClientProxy`. If a particular `ClientProxy`
        is not included in this list, it means that this `ClientProxy`
        will not participate in the next round of federated learning.
        """


    def aggregate_fit_eval(self,
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
            Successful updates from the previously selected and configured
            clients. Each pair of `(ClientProxy, FitRes)` constitutes a
            successful update from one of the previously selected clients. Not
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
