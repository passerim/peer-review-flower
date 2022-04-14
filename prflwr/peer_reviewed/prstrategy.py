from abc import abstractmethod
from typing import Dict, List, Optional, Tuple, Union
from overrides import overrides

from flwr.common import (
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    Weights
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from prmultrev import MultipleReviewStrategy


class PeerReviewStrategy(MultipleReviewStrategy):

    @overrides
    def configure_fit(
        self, 
        rnd: int, 
        parameters: Parameters, 
        client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        return self.configure_train(rnd, parameters, client_manager)

    @overrides
    def aggregate_fit(
        self, 
        rnd: int, 
        results: List[Tuple[ClientProxy, FitRes]], 
        failures: List[BaseException]
    ) -> Union[Tuple[Optional[Parameters], Dict[str, Scalar]], Optional[Weights]]:
        return self.aggregate_train(rnd, results, failures).pop()
