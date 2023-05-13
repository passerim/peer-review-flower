import unittest
from typing import Callable, List, Optional
from unittest.mock import MagicMock, Mock

import numpy as np
from flwr.common import (
    Code,
    DisconnectRes,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    ReconnectIns,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from overrides import overrides

from prflwr.peer_review import (
    PeerReviewNumPyClient,
    PeerReviewServer,
    PrConfig,
    ReviewIns,
    TrainIns,
)
from prflwr.peer_review.strategy import PeerReviewStrategy
from tests.unit.test_strategy import FailingStrategy

OK_STATUS = Status(code=Code.OK, message="")
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
    def __init__(self, cid: str, client: PeerReviewNumPyClient):
        super().__init__(cid)
        self.client: PeerReviewNumPyClient = client

    @overrides
    def get_properties(
        self, ins: GetPropertiesIns, timeout: Optional[float]
    ) -> GetPropertiesRes:
        # This method is not expected to be called
        raise NotImplementedError

    @overrides
    def get_parameters(
        self, ins: GetParametersIns, timeout: Optional[float]
    ) -> GetParametersRes:
        return GetParametersRes(
            status=OK_STATUS,
            parameters=ndarrays_to_parameters(self.client.get_parameters({})),
        )

    @overrides
    def fit(self, ins: FitIns, timeout: Optional[float]) -> FitRes:
        is_review = ins.config.get(PrConfig.REVIEW_FLAG)
        if is_review:
            return FitRes(
                status=OK_STATUS,
                *self.client.review(parameters_to_ndarrays(ins.parameters), ins.config)
            )
        else:
            return FitRes(
                status=OK_STATUS,
                *self.client.train(parameters_to_ndarrays(ins.parameters), ins.config)
            )

    @overrides
    def evaluate(self, ins: EvaluateIns, timeout: Optional[float]) -> EvaluateRes:
        return EvaluateRes(
            status=OK_STATUS,
            *self.client.evaluate(parameters_to_ndarrays(ins.parameters), ins.config)
        )

    @overrides
    def reconnect(self, ins: ReconnectIns, timeout: Optional[float]) -> DisconnectRes:
        # This method is not expected to be called
        raise NotImplementedError


def get_parameters_mock() -> MagicMock:
    return MagicMock(return_value=[TEST_ARRAY])


def train_mock() -> MagicMock:
    return MagicMock(
        return_value=(
            [TEST_ARRAY],
            0,
            {PrConfig.REVIEW_FLAG: False},
        )
    )


def review_mock() -> MagicMock:
    return MagicMock(
        return_value=(
            [TEST_ARRAY],
            0,
            {PrConfig.REVIEW_FLAG: True},
        )
    )


def evaluate_mock() -> MagicMock:
    return MagicMock(return_value=(1, 0, {}))


def failing_client():
    client = MagicMock(spec=PeerReviewNumPyClient)
    client.get_parameters = MagicMock(side_effect=Exception)
    client.train = MagicMock(side_effect=Exception)
    client.review = MagicMock(side_effect=Exception)
    client.evaluate = MagicMock(side_effect=Exception)
    return ClientProxyAdapter(TEST_CID, client)


def failing_train_client():
    client = MagicMock(spec=PeerReviewNumPyClient)
    client.get_parameters = get_parameters_mock()
    client.train = MagicMock(side_effect=Exception)
    client.review = MagicMock(side_effect=Exception)
    client.evaluate = MagicMock(side_effect=Exception)
    return ClientProxyAdapter(TEST_CID, client)


def failing_review_client():
    client = MagicMock(spec=PeerReviewNumPyClient)
    client.get_parameters = get_parameters_mock()
    client.train = train_mock()
    client.review = MagicMock(side_effect=Exception)
    client.evaluate = MagicMock(side_effect=NotImplementedError)
    return ClientProxyAdapter(TEST_CID, client)


def failing_evaluate_client():
    client = MagicMock(spec=PeerReviewNumPyClient)
    client.get_parameters = get_parameters_mock()
    client.train = train_mock()
    client.review = review_mock()
    client.evaluate = MagicMock(side_effect=NotImplementedError)
    return ClientProxyAdapter(TEST_CID, client)


def successful_client():
    client = MagicMock(spec=PeerReviewNumPyClient)
    client.get_parameters = get_parameters_mock()
    client.train = train_mock()
    client.review = review_mock()
    client.evaluate = evaluate_mock()
    return ClientProxyAdapter(TEST_CID, client)


def configure_train_mock(clients: List[ClientProxy]):
    return MagicMock(
        return_value=[
            (
                client,
                TrainIns(
                    parameters=ndarrays_to_parameters([TEST_ARRAY]),
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
                ReviewIns(
                    parameters=ndarrays_to_parameters([TEST_ARRAY]),
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
                EvaluateIns(parameters=ndarrays_to_parameters([TEST_ARRAY]), config={}),
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
    def stop_review(server_round, review_round, *args):
        if review_round >= review_rounds:
            return True if review_round >= review_rounds else False

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
            PeerReviewServer.is_parameters_type(ndarrays_to_parameters([TEST_ARRAY]))
        )
        self.assertFalse(PeerReviewServer.is_weights_type(TEST_ARRAY))
        self.assertFalse(PeerReviewServer.is_parameters_type([i for i in range(10)]))
        self.assertFalse(PeerReviewServer.is_parameters_type("TEST_ARRAY"))

    def test_check_train(self):
        results = [
            (
                successful_client(),
                FitRes(
                    status=OK_STATUS,
                    parameters=None,
                    num_examples=0,
                    metrics={PrConfig.REVIEW_FLAG: False},
                ),
            )
        ]
        failures = [BaseException()]
        results, failures = PeerReviewServer._check_train(results, failures)
        self.assertEqual(len(results), 1)
        self.assertEqual(len(failures), 1)
        results = [
            (
                successful_client(),
                FitRes(
                    status=OK_STATUS,
                    parameters=None,
                    num_examples=0,
                    metrics={PrConfig.REVIEW_FLAG: True},
                ),
            )
        ]
        results, failures = PeerReviewServer._check_train(results, failures)
        self.assertEqual(len(results), 0)
        self.assertEqual(len(failures), 2)

    def test_check_review(self):
        results = [
            (
                successful_client(),
                FitRes(
                    status=OK_STATUS,
                    parameters=None,
                    num_examples=0,
                    metrics={PrConfig.REVIEW_FLAG: True},
                ),
            )
        ]
        failures = [BaseException()]
        results, failures = PeerReviewServer._check_review(results, failures)
        self.assertEqual(len(results), 1)
        self.assertEqual(len(failures), 1)
        results = [
            (
                successful_client(),
                FitRes(
                    status=OK_STATUS,
                    parameters=None,
                    num_examples=0,
                    metrics={PrConfig.REVIEW_FLAG: False},
                ),
            )
        ]
        results, failures = PeerReviewServer._check_review(results, failures)
        self.assertEqual(len(results), 0)
        self.assertEqual(len(failures), 2)


if __name__ == "__main__":
    unittest.main()
