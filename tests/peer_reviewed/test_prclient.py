from typing import Dict, List, Tuple, Union
import unittest

from flwr.common.parameter import ndarray_to_bytes
from flwr.common.typing import Config, Parameters, Properties, Scalar

import numpy as np

from prflwr.peer_reviewed.prclient import PeerReviewClient
from prflwr.peer_reviewed.prconfig import REVIEW_FLAG


class FailingClient(PeerReviewClient):
    """ A client which always fails - raises Exception upon any method call.
    """
    
    def get_properties(self, config: Config) -> Properties:
        raise Exception
    
    def get_parameters(self) -> List[np.ndarray]:
        raise Exception
    
    def train(
        self, 
        parameters: List[np.ndarray], 
        config: Dict[str, Scalar]
    ) -> Tuple[List[np.ndarray], int, Dict[str, Scalar]]:
        raise Exception
    
    def review(
        self, 
        parameters: List[np.ndarray], 
        config: Dict[str, Scalar]
    ) -> Tuple[List[np.ndarray], int, Scalar]:
        raise Exception
    
    def evaluate(
        self, 
        parameters: List[np.ndarray], 
        config: Dict[str, Scalar]
    ) -> Union[
        Tuple[float, int, Dict[str, Scalar]], 
        Tuple[int, float, float], 
        Tuple[int, float, float, Dict[str, Scalar]]
    ]:
        raise Exception


class ClientTrain(FailingClient):
    """ A client with a train method implementation.
    """
    
    def train(
        self, 
        parameters: List[np.ndarray], 
        config: Dict[str, Scalar]
    ) -> Tuple[List[np.ndarray], int, Dict[str, Scalar]]:
        return parameters, 0, {}


class ClientReview(FailingClient):
    """ A client with a review method implementation.
    """
    
    def review(
        self, 
        parameters: List[np.ndarray], 
        config: Dict[str, Scalar]
    ) -> Tuple[List[np.ndarray], int, Scalar]:
        raise Exception


class TestPeerReviewClient(unittest.TestCase):
    """ Tests correct routing of PeerReviewClient fit method calls
        to train and review methods of subclasses based on flag value.
    """
    
    def test_train_success(self):
        client = ClientTrain()
        arr = np.array([[1, 2], [3, 4], [5, 6]])
        parameters = Parameters(tensors=[arr], tensor_type="")
        fit_config = {REVIEW_FLAG: False}
        _, _, train_conf = client.fit(parameters, fit_config)
        assert train_conf[REVIEW_FLAG] == False
    
    def test_train_failure(self):
        client = ClientTrain()
        arr = np.array([[1, 2], [3, 4], [5, 6]])
        parameters = Parameters(tensors=[arr], tensor_type="")
        fit_config = {REVIEW_FLAG: False}
        _, _, train_conf = client.fit(parameters, fit_config)
        try:
            client.fit(parameters, fit_config)
        except Exception as e:
            assert isinstance(e, Exception)
            
    def test_review_success(self):
        client = ClientReview()
        arr = np.array([[1, 2], [3, 4], [5, 6]])
        parameters = Parameters(tensors=[arr], tensor_type="")
        fit_config = {REVIEW_FLAG: True}
        _, _, train_conf = client.fit(parameters, fit_config)
        assert train_conf[REVIEW_FLAG] == True
    
    def test_review_failure(self):
        client = ClientReview()
        arr = np.array([[1, 2], [3, 4], [5, 6]])
        parameters = Parameters(tensors=[arr], tensor_type="")
        fit_config = {REVIEW_FLAG: False}
        _, _, train_conf = client.fit(parameters, fit_config)
        try:
            client.fit(parameters, fit_config)
        except Exception as e:
            assert isinstance(e, Exception)


if __name__ == "__main__":
    unittest.main()
