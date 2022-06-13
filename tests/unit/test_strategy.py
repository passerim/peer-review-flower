import unittest
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock

from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters, Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from prflwr.peer_reviewed.strategy import (
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

    def configure_train(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        raise ConfigureTrainException

    def aggregate_train(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> List[Tuple[Optional[Parameters], Dict[str, Scalar]]]:
        raise AggregateTrainException

    def configure_review(
        self,
        rnd: int,
        review_rnd: int,
        parameters: Parameters,
        client_manager: ClientManager,
        parameters_aggregated: List[Optional[Parameters]],
        metrics_aggregated: List[Dict[str, Scalar]],
    ) -> List[Tuple[ClientProxy, FitIns]]:
        raise ConfigureReviewException

    def aggregate_review(
        self,
        rnd: int,
        review_rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> List[Tuple[Optional[Parameters], Dict[str, Scalar]]]:
        raise AggregateReviewException

    def aggregate_after_review(
        self,
        rnd: int,
        parameters_aggregated: List[Optional[Parameters]],
        metrics_aggregated: List[Dict[str, Scalar]],
        parameters: Optional[Parameters] = None,
    ) -> Optional[Parameters]:
        raise AggregateAfterReviewException

    def stop_review(
        self,
        rnd: int,
        review_rnd: int,
        parameters: Parameters,
        client_manager: ClientManager,
        parameters_aggregated: List[Optional[Parameters]],
        metrics_aggregated: List[Dict[str, Scalar]],
    ) -> bool:
        raise StopReviewException

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        raise InitializeParametersException

    def configure_evaluate(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        raise ConfigureEvaluateException

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        raise AggregateEvaluateException

    def evaluate(
        self, parameters: Parameters
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
        res, _ = self.strategy.aggregate_train(None, None, None)
        self.assertIsNone(res)

    def test_aggregate_review(self):
        res, _ = self.strategy.aggregate_review(None, None, None, None)
        self.assertIsNone(res)

    def test_aggregate_evaluate(self):
        res, _ = self.strategy.aggregate_evaluate(None, None, None)
        self.assertIsNone(res)

    def test_stop_review(self):
        res = self.strategy.stop_review(None, None, None, None, None, None)
        self.assertTrue(res)

    def test_evaluate(self):
        res = self.strategy.evaluate(None)
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
