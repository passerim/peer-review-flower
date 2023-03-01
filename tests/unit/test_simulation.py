import unittest
from logging import INFO
from unittest.mock import MagicMock

import numpy as np
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    Parameters,
    Status,
    ndarray_to_bytes,
)
from flwr.common.logger import log
from flwr.server import ServerConfig, SimpleClientManager
from overrides import overrides

from prflwr.peer_review.client import PeerReviewClient
from prflwr.peer_review.server import PeerReviewServer
from prflwr.peer_review.strategy import PeerReviewedFedAvg, PeerReviewStrategy
from prflwr.peer_review.typing import ReviewIns, ReviewRes, TrainIns, TrainRes
from prflwr.simulation import start_simulation

OK_STATUS = Status(Code.OK, "")
PROXY_CALLS = ["get_parameters", "fit", "fit", "evaluate"]


class NamedSimulationClient(PeerReviewClient):
    def __init__(self, cid: int):
        log(INFO, "[Client %d] Initialized." % cid)
        arr = np.array([[1, 2], [3, 4], [5, 6]])
        arr_serialized = ndarray_to_bytes(arr)
        self.parameters = Parameters(tensors=[arr_serialized], tensor_type="")

    @overrides
    def get_properties(self, ins: GetPropertiesIns) -> GetPropertiesRes:
        # This method is not expected to be called
        raise Exception

    @overrides
    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        return GetParametersRes(OK_STATUS, self.parameters)

    @overrides
    def train(self, ins: TrainIns) -> TrainRes:
        return TrainRes(
            OK_STATUS, self.get_parameters(GetParametersIns({})).parameters, 1, {}
        )

    @overrides
    def review(self, ins: ReviewIns) -> ReviewRes:
        return ReviewRes(
            OK_STATUS, self.get_parameters(GetParametersIns({})).parameters, 1, {}
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        return EvaluateRes(OK_STATUS, 0.0, 1, {})


def client_fn(cid: str):
    client = NamedSimulationClient(int(cid))
    return MagicMock(wraps=client)


class TestSimulationWithPrServer(unittest.TestCase):
    def test_server_is_peer_reviewed(self):
        # Define strategy and assert it is a subclass of PeerReviewStrategy
        strategy = PeerReviewedFedAvg(
            min_fit_clients=1,
            min_review_clients=1,
            min_evaluate_clients=1,
            min_available_clients=1,
        )
        self.assertIsInstance(strategy, PeerReviewStrategy)

        # Define server and assert it is a subclass of PeerReviewServer
        client_manager = SimpleClientManager()
        server = PeerReviewServer(client_manager=client_manager, strategy=strategy)
        self.assertIsInstance(server, PeerReviewServer)

        # Start simulation and assert a value for hist is actually returned
        hist = start_simulation(
            client_fn=client_fn,
            num_clients=1,
            config=ServerConfig(num_rounds=1),
            strategy=strategy,
            server=server,
            client_manager=client_manager,
        )
        self.assertIsNotNone(hist)

        # Test that calls to the proxy are forwarded to the client
        self.assertEqual(
            PROXY_CALLS,
            [
                call[0]
                for call in list(client_manager.clients.values())
                .pop()
                .client.mock_calls
            ],
        )


if __name__ == "__main__":
    unittest.main()
