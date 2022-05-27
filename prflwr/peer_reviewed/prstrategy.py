from abc import ABC
from typing import Dict, List, Optional, Tuple

from flwr.common import FitIns, FitRes, Parameters, Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from overrides import overrides

from .prmultrev import MultipleReviewStrategy


class PeerReviewStrategy(MultipleReviewStrategy, ABC):
    """Abstract class to extend implementing methods that define a federated
    learning strategy with support to performing multiple review rounds.
    """

    @overrides
    def configure_fit(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        return self.configure_train(rnd, parameters, client_manager)

    @overrides
    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        return self.aggregate_train(rnd, results, failures).pop()
