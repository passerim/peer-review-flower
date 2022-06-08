from abc import ABC
from functools import wraps
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
                    return []
                elif isinstance(e, ConfigureReviewException):
                    return []
                elif isinstance(e, ConfigureEvaluationException):
                    return []
                elif isinstance(e, AggregateTrainException):
                    return None, {}
                elif isinstance(e, AggregateReviewException):
                    return None, {}
                elif isinstance(e, AggregateEvaluationException):
                    return None, {}
                elif isinstance(e, AggregateAfterReviewException):
                    return None
                elif isinstance(e, EvaluationException):
                    return None
                elif isinstance(e, StopReviewException):
                    return True
                elif isinstance(e, InitializeParametersException):
                    return None

        return handle

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


class StrategyException(Exception):
    def __init__(self, message):
        super().__init__(message)


class ConfigureTrainException(StrategyException):
    ...


class ConfigureEvaluationException(StrategyException):
    ...


class ConfigureReviewException(StrategyException):
    ...


class AggregateTrainException(StrategyException):
    ...


class AggregateEvaluationException(StrategyException):
    ...


class AggregateReviewException(StrategyException):
    ...


class AggregateAfterReviewException(StrategyException):
    ...


class StopReviewException(StrategyException):
    ...


class EvaluationException(StrategyException):
    ...


class InitializeParametersException(StrategyException):
    ...
