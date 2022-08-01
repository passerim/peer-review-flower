from abc import ABC
from functools import wraps
from typing import Dict, List, Optional, Tuple, Union

from flwr.common import FitIns, FitRes, Parameters, Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from overrides import overrides
from prflwr.peer_review.strategy.exceptions import (
    AggregateAfterReviewException,
    AggregateEvaluateException,
    AggregateReviewException,
    AggregateTrainException,
    ConfigureEvaluateException,
    ConfigureReviewException,
    ConfigureTrainException,
    EvaluateException,
    InitializeParametersException,
    StopReviewException,
)
from prflwr.peer_review.strategy.prmultrev import MultipleReviewStrategy


class PeerReviewStrategy(MultipleReviewStrategy, ABC):
    """Abstract class to extend implementing methods that define a federated
    learning strategy with support to performing multiple review rounds.
    """

    __strategy_methods = [
        "configure_fit",
        "aggregate_fit",
        "configure_train",
        "aggregate_train",
        "configure_review",
        "aggregate_review",
        "aggregate_after_review",
        "stop_review",
        "initialize_parameters",
        "configure_evaluate",
        "aggregate_evaluate",
        "evaluate",
    ]

    def __init__(self):
        for attr in dir(self):
            method = self.__getattribute__(attr)
            if callable(method) and attr in self.__strategy_methods:
                setattr(self, attr, self.handle_exceptions(method))

    @staticmethod
    def handle_exceptions(method):
        @wraps(method)
        def handle(*args, **kwargs):
            try:
                return method(*args, **kwargs)
            except Exception as e:
                if isinstance(e, ConfigureTrainException):
                    return None
                elif isinstance(e, ConfigureReviewException):
                    return None
                elif isinstance(e, ConfigureEvaluateException):
                    return None
                elif isinstance(e, AggregateTrainException):
                    return None, {}
                elif isinstance(e, AggregateReviewException):
                    return None, {}
                elif isinstance(e, AggregateEvaluateException):
                    return None, {}
                elif isinstance(e, AggregateAfterReviewException):
                    return None
                elif isinstance(e, EvaluateException):
                    return None
                elif isinstance(e, StopReviewException):
                    return True
                elif isinstance(e, InitializeParametersException):
                    return None
                else:
                    raise TypeError

        return handle

    @overrides
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        return self.configure_train(server_round, parameters, client_manager)

    @overrides
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        return self.aggregate_train(server_round, results, failures).pop()
