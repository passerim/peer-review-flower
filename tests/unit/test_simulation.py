import unittest

import numpy as np
from flwr.client import Client
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    Parameters,
    Status,
    ndarray_to_bytes,
)
from flwr.server import ServerConfig, SimpleClientManager
from overrides import overrides

from prflwr.peer_review import PeerReviewServer
from prflwr.peer_review.strategy import PeerReviewedFedAvg, PeerReviewStrategy
from prflwr.simulation import start_simulation

OK_STATUS = Status(Code.OK, "")


class NamedSimulationClient(Client):
    def __init__(self):
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
    def fit(self, ins: FitIns) -> FitRes:
        return FitRes(
            OK_STATUS, self.get_parameters(GetParametersIns({})).parameters, 0, {}
        )

    @overrides
    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        # This method is not expected to be called
        raise Exception


class TestSimulationWithPrServer(unittest.TestCase):
    def test_server_is_peer_reviewed(self):
        # Define strategy and assert it is a subclass of PeerReviewStrategy
        strategy = PeerReviewedFedAvg(
            fraction_review=1.0,
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=2,
            min_evaluate_clients=2,
            min_available_clients=2,
        )
        self.assertIsInstance(strategy, PeerReviewStrategy)

        # Define server and assert it is a subclass of PeerReviewServer
        client_manager = SimpleClientManager()
        server = PeerReviewServer(client_manager=client_manager, strategy=strategy)
        self.assertIsInstance(server, PeerReviewServer)

        # Start simulation and assert a value for hist is actually returned
        hist = start_simulation(
            client_fn=lambda cid: NamedSimulationClient(),
            num_clients=1,
            config=ServerConfig(num_rounds=0),
            strategy=strategy,
            server=server,
            client_manager=client_manager,
        )
        self.assertIsNotNone(hist)


if __name__ == "__main__":
    unittest.main()
