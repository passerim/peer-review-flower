import unittest
from typing import Dict, List, Optional, Tuple, Union
from unittest.mock import MagicMock

from flwr.common import EvaluateIns, EvaluateRes, Parameters, Scalar
from flwr.server import ClientManager
from flwr.server.client_proxy import ClientProxy
from overrides import overrides

from prflwr.peer_review import ReviewIns, ReviewRes, TrainIns, TrainRes
from prflwr.peer_review.strategy import (
    AggregateAfterReviewException,
    AggregateEvaluateException,
    AggregateReviewException,
    AggregateTrainException,
    ConfigureEvaluateException,
    ConfigureReviewException,
    ConfigureTrainException,
    EvaluateException,
    InitializeParametersException,
    PeerReviewStrategy,
    StopReviewException,
)


class FailingStrategy(PeerReviewStrategy):
    def __init__(self):
        super(FailingStrategy, self).__init__()

    @overrides
    def configure_train(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, TrainIns]]:
        raise ConfigureTrainException

    @overrides
    def aggregate_train(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, TrainRes]],
        failures: List[Union[Tuple[ClientProxy, TrainRes], BaseException]],
        parameters: Optional[Parameters] = None,
    ) -> List[Tuple[Optional[Parameters], Dict[str, Scalar]]]:
        raise AggregateTrainException

    @overrides
    def configure_review(
        self,
        server_round: int,
        review_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
        parameters_aggregated: List[Optional[Parameters]],
        metrics_aggregated: List[Dict[str, Scalar]],
    ) -> List[Tuple[ClientProxy, ReviewIns]]:
        raise ConfigureReviewException

    @overrides
    def aggregate_review(
        self,
        server_round: int,
        review_round: int,
        results: List[Tuple[ClientProxy, ReviewRes]],
        failures: List[Union[Tuple[ClientProxy, ReviewRes], BaseException]],
        parameters: Parameters,
        parameters_aggregated: List[Optional[Parameters]],
        metrics_aggregated: List[Dict[str, Scalar]],
    ) -> List[Tuple[Optional[Parameters], Dict[str, Scalar]]]:
        raise AggregateReviewException

    @overrides
    def aggregate_after_review(
        self,
        server_round: int,
        parameters: Parameters,
        parameters_aggregated: List[Optional[Parameters]],
        metrics_aggregated: List[Dict[str, Scalar]],
    ) -> Optional[Parameters]:
        raise AggregateAfterReviewException

    @overrides
    def stop_review(
        self,
        server_round: int,
        review_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
        parameters_aggregated: List[Optional[Parameters]],
        metrics_aggregated: List[Dict[str, Scalar]],
    ) -> bool:
        raise StopReviewException

    @overrides
    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        raise InitializeParametersException

    @overrides
    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        raise ConfigureEvaluateException

    @overrides
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        raise AggregateEvaluateException

    @overrides
    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        raise EvaluateException


class TestFailingPeerReviewStrategy(unittest.TestCase):
    def setUp(self) -> None:
        self.strategy: PeerReviewStrategy = FailingStrategy()

    def test_configure_train(self):
        res = self.strategy.configure_train(None, None, None)
        self.assertIsNone(res)

    def test_configure_review(self):
        res = self.strategy.configure_review(None, None, None, None, None, None)
        self.assertIsNone(res)

    def test_configure_evaluate(self):
        res = self.strategy.configure_evaluate(None, None, None)
        self.assertIsNone(res)

    def test_aggregate_train(self):
        res, _ = self.strategy.aggregate_train(None, None, None, None)
        self.assertIsNone(res)

    def test_aggregate_review(self):
        res, _ = self.strategy.aggregate_review(
            None, None, None, None, None, None, None
        )
        self.assertIsNone(res)

    def test_aggregate_evaluate(self):
        res, _ = self.strategy.aggregate_evaluate(None, None, None)
        self.assertIsNone(res)

    def test_stop_review(self):
        res = self.strategy.stop_review(None, None, None, None, None, None)
        self.assertTrue(res)

    def test_evaluate(self):
        res = self.strategy.evaluate(None, None)
        self.assertIsNone(res)

    def test_aggregate_after_review(self):
        res = self.strategy.aggregate_after_review(None, None, None, None)
        self.assertIsNone(res)

    def test_initialize_parameters(self):
        res = self.strategy.initialize_parameters(None)
        self.assertIsNone(res)


class MockStrategy(FailingStrategy):
    def __init__(self):
        super().__init__()
        self.configure_train = MagicMock()
        self.configure_review = MagicMock()
        self.configure_evaluate = MagicMock()
        self.aggregate_train = MagicMock()
        self.aggregate_review = MagicMock()
        self.aggregate_evaluate = MagicMock()
        self.aggregate_after_review = MagicMock()
        self.stop_review = MagicMock()
        self.initialize_parameters = MagicMock()
        self.evaluate = MagicMock()


class TestSubstitution(unittest.TestCase):
    def setUp(self) -> None:
        self.strategy: PeerReviewStrategy = MockStrategy()

    def test_configure_fit_substitution(self):
        self.strategy.configure_fit(None, None, None)
        self.strategy.configure_train.assert_called_once()

    def test_aggregate_fit_substitution(self):
        self.strategy.aggregate_fit(None, None, None)
        self.strategy.aggregate_train.assert_called_once()


if __name__ == "__main__":
    unittest.main()
