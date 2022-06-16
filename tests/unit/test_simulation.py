import unittest

import numpy as np
from flwr.common import (
    Disconnect,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    ParametersRes,
    PropertiesIns,
    PropertiesRes,
    Reconnect,
    ndarray_to_bytes,
)
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from prflwr.peer_review import PeerReviewServer
from prflwr.peer_review.strategy import PeerReviewedFedAvg, PeerReviewStrategy
from prflwr.simulation import start_simulation


class NamedSimulationClient(ClientProxy):
    def __init__(self, cid: str):
        super().__init__(cid)
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


class TestSimulationWithPrServer(unittest.TestCase):
    def test_server_is_peer_reviewed(self):

        # Define strategy and assert it is a subclass of PeerReviewStrategy
        strategy = PeerReviewedFedAvg(
            fraction_review=1.0,
            fraction_fit=1.0,
            fraction_eval=1.0,
            min_fit_clients=2,
            min_eval_clients=2,
            min_available_clients=2,
        )
        self.assertIsInstance(strategy, PeerReviewStrategy)

        # Define server and assert it is a sublass of PeerReviewServer
        client_manager = SimpleClientManager()
        server = PeerReviewServer(client_manager=client_manager, strategy=strategy)
        self.assertIsInstance(server, PeerReviewServer)

        # Start simulation and assert a value for hist is actually returned
        hist = start_simulation(
            client_fn=lambda cid: NamedSimulationClient(cid),
            num_clients=1,
            num_rounds=0,
            strategy=strategy,
            server=server,
            client_manager=client_manager,
            client_resources={"num_cpus": 1, "num_gpus": 0},
            ray_init_args={"local_mode": True, "include_dashboard": False},
        )
        self.assertIsNotNone(hist)


if __name__ == "__main__":
    unittest.main()
