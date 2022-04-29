import unittest

import numpy as np
from examples.peer_reviewed.strategy import PeerReviewedFedAvg
from flwr.common import (Disconnect, EvaluateIns, EvaluateRes, FitIns, FitRes,
                         Parameters, ParametersRes, PropertiesIns,
                         PropertiesRes, Reconnect, ndarray_to_bytes)
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from prflwr.peer_reviewed.prserver import PeerReviewServer
from prflwr.peer_reviewed.prstrategy import PeerReviewStrategy
from prflwr.simulation.app import start_simulation


class NamedSimulationClient(ClientProxy):

    def __init__(self, cid: str):
        super().init(cid)
        arr = np.array([[1, 2], [3, 4], [5, 6]])
        arr_serialized = ndarray_to_bytes(arr)
        self.parameters = Parameters(tensors=[arr_serialized], tensor_type="")

    def get_properties(self, ins: PropertiesIns) -> PropertiesRes:
        # This method is not expected to be called
        raise Exception

    def get_parameters(self) -> ParametersRes:
        return ParametersRes(self.parameters)

    def fit(self, ins: FitIns) -> FitRes:
        return self.get_parameters()

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        # This method is not expected to be called
        raise Exception

    def reconnect(self, reconnect: Reconnect) -> Disconnect:
        # This method is not expected to be called
        raise Exception


class TestCentralizedTraining(unittest.TestCase):

    def test_server_is_peer_reviewed(self):

        # Define strategy and assert it is a sublass of PeerReviewStrategy
        strategy = PeerReviewedFedAvg(
            fraction_review=1.0,
            fraction_fit=1.0,
            fraction_eval=1.0,
            min_fit_clients=2,
            min_eval_clients=2,
            min_available_clients=2,
        )
        assert isinstance(strategy, PeerReviewStrategy)

        # Define server and assert it is a sublass of PeerReviewServer
        server = PeerReviewServer(
            client_manager=SimpleClientManager(), strategy=strategy
        )
        assert isinstance(server, PeerReviewServer)

        # Start simulation and assert a value for hist is actually returned
        hist = start_simulation(
            client_fn=lambda cid: NamedSimulationClient(cid),
            num_clients=1,
            num_rounds=0,
            strategy=strategy,
            server=server
        )
        assert hist is not None


if __name__ == "__main__":
    unittest.main()
