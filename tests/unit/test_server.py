import unittest
from typing import Callable, List, Optional
from unittest.mock import MagicMock, Mock

import numpy as np
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    ParametersRes,
    PropertiesIns,
    PropertiesRes,
    parameters_to_weights,
    weights_to_parameters,
)
from flwr.common.typing import Disconnect, Reconnect
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.fedavg import FedAvg
from prflwr.peer_review import PeerReviewClient, PeerReviewServer, PrConfig
from prflwr.peer_review.strategy import PeerReviewStrategy
from tests.unit.test_strategy import FailingStrategy

TEST_VALUE = 42
TEST_ROUNDS = 1
TEST_CID = "0"
TEST_ARRAY = np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
SUCCESSFUL_STRATEGY_ROUND = [
    "initialize_parameters",
    "evaluate",
    "configure_train",
    "aggregate_train",
    "configure_review",
    "aggregate_review",
    "stop_review",
    "aggregate_after_review",
    "evaluate",
    "configure_evaluate",
    "aggregate_evaluate",
]


class TestPeerReviewServerConstructor(unittest.TestCase):
    def test_constructor_max_workers(self):
        self.server = PeerReviewServer(
            SimpleClientManager(), Mock(spec=PeerReviewStrategy), max_workers=TEST_VALUE
        )
        self.assertEqual(self.server.max_workers, TEST_VALUE)

    def test_constructor_review_rounds(self):
        self.server = PeerReviewServer(
            SimpleClientManager(),
            Mock(spec=PeerReviewStrategy),
            max_review_rounds=TEST_VALUE,
        )
        self.assertEqual(self.server.max_review_rounds, TEST_VALUE)

    def test_constructor_max_everything(self):
        self.server = PeerReviewServer(
            SimpleClientManager(),
            Mock(spec=PeerReviewStrategy),
            max_workers=TEST_VALUE,
            max_review_rounds=TEST_VALUE,
        )
        self.assertEqual(self.server.max_workers, TEST_VALUE)
        self.assertEqual(self.server.max_review_rounds, TEST_VALUE)

    def test_constructor_strategy(self):
        self.server = PeerReviewServer(SimpleClientManager(), FedAvg())
        self.assertEqual(self.server.fit, super(self.server.__class__, self.server).fit)
        self.assertEqual(
            self.server.fit_round, super(self.server.__class__, self.server).fit_round
        )


class ClientProxyAdapter(ClientProxy):
    def __init__(self, cid: str, client: PeerReviewClient):
        super().__init__(cid)
        self.client: PeerReviewClient = client

    def get_properties(
        self, ins: PropertiesIns, timeout: Optional[float]
    ) -> PropertiesRes:
        raise NotImplementedError

    def get_parameters(self, timeout: Optional[float]) -> ParametersRes:
        return self.client.get_parameters()

    def fit(self, ins: FitIns, timeout: Optional[float]) -> FitRes:
        is_review = ins.config.get(PrConfig.REVIEW_FLAG)
        if is_review:
            return self.client.review(parameters_to_weights(ins.parameters), ins.config)
        else:
            return self.client.train(parameters_to_weights(ins.parameters), ins.config)

    def evaluate(self, ins: EvaluateIns, timeout: Optional[float]) -> EvaluateRes:
        return self.client.evaluate(parameters_to_weights(ins.parameters), ins.config)

    def reconnect(self, reconnect: Reconnect, timeout: Optional[float]) -> Disconnect:
        raise NotImplementedError


def failing_client():
    client = MagicMock(spec=PeerReviewClient)
    client.get_parameters = MagicMock(side_effect=Exception)
    client.train = MagicMock(side_effect=Exception)
    client.review = MagicMock(side_effect=Exception)
    client.evaluate = MagicMock(side_effect=Exception)
    return ClientProxyAdapter(TEST_CID, client)


def failing_train_client():
    client = MagicMock(spec=PeerReviewClient)
    client.get_parameters = MagicMock(
        return_value=ParametersRes(parameters=weights_to_parameters([TEST_ARRAY]))
    )
    client.train = MagicMock(side_effect=Exception)
    client.review = MagicMock(side_effect=Exception)
    client.evaluate = MagicMock(side_effect=Exception)
    return ClientProxyAdapter(TEST_CID, client)


def failing_review_client():
    client = MagicMock(spec=PeerReviewClient)
    client.get_parameters = MagicMock(
        return_value=ParametersRes(parameters=weights_to_parameters([TEST_ARRAY]))
    )
    client.train = MagicMock(
        return_value=FitRes(
            parameters=weights_to_parameters([TEST_ARRAY]),
            num_examples=0,
            metrics={PrConfig.REVIEW_FLAG: False},
        )
    )
    client.review = MagicMock(side_effect=Exception)
    client.evaluate = MagicMock(side_effect=NotImplementedError)
    return ClientProxyAdapter(TEST_CID, client)


def failing_evaluate_client():
    client = MagicMock(spec=PeerReviewClient)
    client.get_parameters = MagicMock(
        return_value=ParametersRes(parameters=weights_to_parameters([TEST_ARRAY]))
    )
    client.train = MagicMock(
        return_value=FitRes(
            parameters=weights_to_parameters([TEST_ARRAY]),
            num_examples=0,
            metrics={PrConfig.REVIEW_FLAG: False},
        )
    )
    client.review = MagicMock(
        return_value=FitRes(
            parameters=weights_to_parameters([TEST_ARRAY]),
            num_examples=0,
            metrics={PrConfig.REVIEW_FLAG: True},
        )
    )
    client.evaluate = MagicMock(side_effect=NotImplementedError)
    return ClientProxyAdapter(TEST_CID, client)


def successful_client():
    client = MagicMock(spec=PeerReviewClient)
    client.get_parameters = MagicMock(
        return_value=ParametersRes(parameters=weights_to_parameters([TEST_ARRAY]))
    )
    client.train = MagicMock(
        return_value=FitRes(
            parameters=weights_to_parameters([TEST_ARRAY]),
            num_examples=0,
            metrics={PrConfig.REVIEW_FLAG: False},
        )
    )
    client.review = MagicMock(
        return_value=FitRes(
            parameters=weights_to_parameters([TEST_ARRAY]),
            num_examples=0,
            metrics={PrConfig.REVIEW_FLAG: True},
        )
    )
    client.evaluate = MagicMock(
        return_value=EvaluateRes(
            loss=1,
            num_examples=0,
            metrics={},
        )
    )
    return ClientProxyAdapter(TEST_CID, client)


def configure_train_mock(clients: List[ClientProxy]):
    return MagicMock(
        return_value=[
            (
                client,
                FitIns(
                    parameters=weights_to_parameters([TEST_ARRAY]),
                    config={PrConfig.REVIEW_FLAG: False},
                ),
            )
            for client in clients
        ]
    )


def configure_review_mock(clients: List[ClientProxy]):
    return MagicMock(
        return_value=[
            (
                client,
                FitIns(
                    parameters=weights_to_parameters([TEST_ARRAY]),
                    config={PrConfig.REVIEW_FLAG: True},
                ),
            )
            for client in clients
        ]
    )


def configure_evaluate_mock(clients: List[ClientProxy]):
    return MagicMock(
        return_value=[
            (
                client,
                EvaluateIns(parameters=weights_to_parameters([TEST_ARRAY]), config={}),
            )
            for client in clients
        ]
    )


def failing_strategy(*args):
    strategy = MagicMock(spec=FailingStrategy)
    strategy.configure_train = MagicMock(return_value=None)
    strategy.configure_review = MagicMock(return_value=None)
    strategy.configure_evaluate = MagicMock(return_value=None)
    strategy.aggregate_train = MagicMock(return_value=(None, {}))
    strategy.aggregate_review = MagicMock(return_value=(None, {}))
    strategy.aggregate_evaluate = MagicMock(return_value=(None, {}))
    strategy.aggregate_after_review = MagicMock(return_value=None)
    strategy.stop_review = MagicMock(return_value=True)
    strategy.initialize_parameters = MagicMock(return_value=None)
    strategy.evaluate = MagicMock(return_value=None)
    return strategy


def failing_review_strategy(clients: List[ClientProxy] = None, *args):
    if clients is None:
        clients = []
    strategy = MagicMock(spec=PeerReviewStrategy)
    strategy.configure_train = configure_train_mock(clients)
    strategy.configure_review = MagicMock(return_value=None)
    strategy.configure_evaluate = MagicMock(return_value=None)
    strategy.aggregate_train = MagicMock(return_value=[([TEST_ARRAY], {})])
    strategy.aggregate_review = MagicMock(return_value=(None, {}))
    strategy.aggregate_evaluate = MagicMock(return_value=(None, {}))
    strategy.aggregate_after_review = MagicMock(return_value=None)
    strategy.stop_review = MagicMock(return_value=True)
    strategy.initialize_parameters = MagicMock(
        return_value=None
    )  # Let the server ask clients for parameters
    strategy.evaluate = MagicMock(return_value=None)
    return strategy


def failing_evaluate_strategy(clients: List[ClientProxy] = None, *args):
    if clients is None:
        clients = []
    strategy = Mock(spec=PeerReviewStrategy)
    strategy.configure_train = configure_train_mock(clients)
    strategy.configure_review = configure_review_mock(clients)
    strategy.configure_evaluate = MagicMock(return_value=None)
    strategy.aggregate_train = MagicMock(return_value=[([TEST_ARRAY], {})])
    strategy.aggregate_review = MagicMock(return_value=[([TEST_ARRAY], {})])
    strategy.aggregate_evaluate = MagicMock(return_value=(None, {}))
    strategy.aggregate_after_review = MagicMock(return_value=None)
    strategy.stop_review = MagicMock(return_value=True)
    strategy.initialize_parameters = MagicMock(
        return_value=None
    )  # Let the server ask clients for parameters
    strategy.evaluate = MagicMock(return_value=None)
    return strategy


def successful_strategy(clients: List[ClientProxy] = None, review_rounds: int = 1):
    def stop_review(rnd, review_rnd, *args):
        if review_rnd >= review_rounds:
            return True if review_rnd >= review_rounds else False

    if clients is None:
        clients = []
    strategy = Mock(spec=PeerReviewStrategy)
    strategy.configure_train = configure_train_mock(clients)
    strategy.configure_review = configure_review_mock(clients)
    strategy.configure_evaluate = configure_evaluate_mock(clients)
    strategy.aggregate_train = MagicMock(return_value=[([TEST_ARRAY], {})])
    strategy.aggregate_review = MagicMock(return_value=[([TEST_ARRAY], {})])
    strategy.aggregate_evaluate = MagicMock(return_value=(0.0, {}))
    strategy.aggregate_after_review = MagicMock(return_value=[TEST_ARRAY])
    strategy.stop_review = MagicMock(side_effect=stop_review)
    strategy.initialize_parameters = MagicMock(
        return_value=None
    )  # Let the server ask clients for parameters
    strategy.evaluate = MagicMock(return_value=(0.0, {}))
    return strategy


class TestPeerReviewServer(unittest.TestCase):
    def custom_setup(
        self,
        strategy_fn: Callable[
            [Optional[List[ClientProxy]], Optional[int]], PeerReviewStrategy
        ],
        client_fn: Callable[[], ClientProxy],
        review_rounds: int = 1,
    ):
        self.client_manager = SimpleClientManager()
        self.client_manager.register(client_fn())
        self.strategy = strategy_fn(
            [proxy for _, proxy in self.client_manager.clients.items()],
            review_rounds,
        )
        self.server = PeerReviewServer(self.client_manager, self.strategy)
        self.server.fit(TEST_ROUNDS, None)

    def get_client(self) -> Optional[ClientProxyAdapter]:
        return next(iter(map(lambda c: c[1], self.client_manager.all().items())), None)

    def test_failing_strategy_and_failing_client(self):
        self.custom_setup(failing_strategy, failing_client)
        client = self.get_client()
        if client:
            client_calls = [call[0] for call in client.client.mock_calls]
            self.assertEqual(client_calls, ["get_parameters"])
        strategy_calls = [call[0] for call in self.strategy.mock_calls]
        self.assertEqual(strategy_calls, ["initialize_parameters"])

    def test_failing_strategy_and_successful_client(self):
        self.custom_setup(failing_strategy, successful_client)
        client = self.get_client()
        if client:
            client_calls = [call[0] for call in client.client.mock_calls]
            self.assertEqual(client_calls, ["get_parameters"])
        strategy_calls = [call[0] for call in self.strategy.mock_calls]
        self.assertEqual(
            strategy_calls,
            SUCCESSFUL_STRATEGY_ROUND[:3],
        )

    def test_failing_review_strategy_and_failing_client(self):
        self.custom_setup(failing_review_strategy, failing_client)
        client = self.get_client()
        if client:
            client_calls = [call[0] for call in client.client.mock_calls]
            self.assertEqual(client_calls, ["get_parameters"])
        strategy_calls = [call[0] for call in self.strategy.mock_calls]
        self.assertEqual(strategy_calls, ["initialize_parameters"])

    def test_failing_review_strategy_and_successful_client(self):
        self.custom_setup(failing_review_strategy, successful_client)
        client = self.get_client()
        if client:
            client_calls = [call[0] for call in client.client.mock_calls]
            self.assertEqual(client_calls, ["get_parameters", "train"])
        strategy_calls = [call[0] for call in self.strategy.mock_calls]
        self.assertTrue("aggregate_review" not in strategy_calls)
        self.assertEqual(
            strategy_calls,
            SUCCESSFUL_STRATEGY_ROUND[:5],
        )

    def test_failing_evaluate_strategy_and_failing_client(self):
        self.custom_setup(failing_evaluate_strategy, failing_client)
        client = self.get_client()
        if client:
            client_calls = [call[0] for call in client.client.mock_calls]
            self.assertEqual(client_calls, ["get_parameters"])
        strategy_calls = [call[0] for call in self.strategy.mock_calls]
        self.assertEqual(strategy_calls, ["initialize_parameters"])

    def test_failing_evaluate_strategy_and_successful_client(self):
        self.custom_setup(failing_evaluate_strategy, successful_client)
        client = self.get_client()
        if client:
            client_calls = [call[0] for call in client.client.mock_calls]
            self.assertEqual(client_calls, ["get_parameters", "train", "review"])
        strategy_calls = [call[0] for call in self.strategy.mock_calls]
        self.assertTrue("aggregate_evaluate" not in strategy_calls)
        self.assertEqual(
            strategy_calls,
            SUCCESSFUL_STRATEGY_ROUND[:-1],
        )

    def test_successful_strategy_and_failing_client(self):
        self.custom_setup(successful_strategy, failing_client)
        client = self.get_client()
        if client:
            client_calls = [call[0] for call in client.client.mock_calls]
            self.assertEqual(client_calls, ["get_parameters"])
        strategy_calls = [call[0] for call in self.strategy.mock_calls]
        self.assertEqual(strategy_calls, ["initialize_parameters"])

    def test_successful_strategy_and_successful_client(self):
        self.custom_setup(successful_strategy, successful_client)
        client = self.get_client()
        if client:
            client_calls = [call[0] for call in client.client.mock_calls]
            self.assertEqual(
                client_calls, ["get_parameters", "train", "review", "evaluate"]
            )
        strategy_calls = [call[0] for call in self.strategy.mock_calls]
        self.assertEqual(strategy_calls, SUCCESSFUL_STRATEGY_ROUND)


class TestPeerReviewServerUtils(unittest.TestCase):
    def test_is_weights_type(self):
        self.assertTrue(PeerReviewServer.is_weights_type([TEST_ARRAY]))
        self.assertFalse(PeerReviewServer.is_weights_type(TEST_ARRAY))
        self.assertFalse(PeerReviewServer.is_weights_type([i for i in range(10)]))
        self.assertFalse(PeerReviewServer.is_weights_type("TEST_ARRAY"))

    def test_is_parameters_type(self):
        self.assertTrue(
            PeerReviewServer.is_parameters_type(weights_to_parameters([TEST_ARRAY]))
        )
        self.assertFalse(PeerReviewServer.is_weights_type(TEST_ARRAY))
        self.assertFalse(PeerReviewServer.is_parameters_type([i for i in range(10)]))
        self.assertFalse(PeerReviewServer.is_parameters_type("TEST_ARRAY"))

    def test_check_train(self):
        results = [
            (
                successful_client(),
                FitRes(
                    parameters=None,
                    num_examples=0,
                    metrics={PrConfig.REVIEW_FLAG: False},
                ),
            )
        ]
        failures = [BaseException]
        results, failures = PeerReviewServer.check_train(results, failures)
        self.assertEqual(len(results), 1)
        self.assertEqual(len(failures), 1)
        results = [
            (
                successful_client(),
                FitRes(
                    parameters=None,
                    num_examples=0,
                    metrics={PrConfig.REVIEW_FLAG: True},
                ),
            )
        ]
        results, failures = PeerReviewServer.check_train(results, failures)
        self.assertEqual(len(results), 0)
        self.assertEqual(len(failures), 2)

    def test_check_review(self):
        results = [
            (
                successful_client(),
                FitRes(
                    parameters=None,
                    num_examples=0,
                    metrics={PrConfig.REVIEW_FLAG: True},
                ),
            )
        ]
        failures = [BaseException]
        results, failures = PeerReviewServer.check_review(results, failures)
        self.assertEqual(len(results), 1)
        self.assertEqual(len(failures), 1)
        results = [
            (
                successful_client(),
                FitRes(
                    parameters=None,
                    num_examples=0,
                    metrics={PrConfig.REVIEW_FLAG: False},
                ),
            )
        ]
        results, failures = PeerReviewServer.check_review(results, failures)
        self.assertEqual(len(results), 0)
        self.assertEqual(len(failures), 2)


if __name__ == "__main__":
    unittest.main()
