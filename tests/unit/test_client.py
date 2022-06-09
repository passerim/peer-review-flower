import unittest
from typing import Dict, List, Tuple, Union

import numpy as np
from flwr.common.typing import Config, Properties, Scalar
from prflwr.peer_reviewed.client import PeerReviewClient
from prflwr.peer_reviewed.config import PrConfig


class FailingClient(PeerReviewClient):
    """A client which always fails - raises Exception upon any method call."""

    def get_properties(self, config: Config) -> Properties:
        raise Exception

    def get_parameters(self) -> List[np.ndarray]:
        raise Exception

    def train(
        self, parameters: List[np.ndarray], config: Dict[str, Scalar]
    ) -> Tuple[List[np.ndarray], int, Dict[str, Scalar]]:
        raise Exception

    def review(
        self, parameters: List[np.ndarray], config: Dict[str, Scalar]
    ) -> Tuple[List[np.ndarray], int, Dict[str, Scalar]]:
        raise Exception

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, Scalar]
    ) -> Union[
        Tuple[float, int, Dict[str, Scalar]],
        Tuple[int, float, float],
        Tuple[int, float, float, Dict[str, Scalar]],
    ]:
        raise Exception


class ClientTrain(FailingClient):
    """A client with a train method implementation."""

    def train(
        self, parameters: List[np.ndarray], config: Dict[str, Scalar]
    ) -> Tuple[List[np.ndarray], int, Dict[str, Scalar]]:
        return parameters, 0, {}


class ClientReview(FailingClient):
    """A client with a review method implementation."""

    def review(
        self, parameters: List[np.ndarray], config: Dict[str, Scalar]
    ) -> Tuple[List[np.ndarray], int, Dict[str, Scalar]]:
        return parameters, 0, {PrConfig.REVIEW_SCORE: 1}


class TestPeerReviewClient(unittest.TestCase):
    """Tests correct routing of PeerReviewClient fit method calls
    to train and review methods of subclasses based on flag value.
    """

    parameters = [np.array([[1, 2], [3, 4], [5, 6]])]

    def test_train_success(self):
        client = ClientTrain()
        fit_config = {PrConfig.REVIEW_FLAG: False}
        _, _, train_conf = client.fit(self.parameters, fit_config)
        self.assertFalse(train_conf[PrConfig.REVIEW_FLAG])

    def test_train_failure(self):
        client = ClientTrain()
        fit_config = {PrConfig.REVIEW_FLAG: False}
        try:
            client.fit(self.parameters, fit_config)
        except Exception as e:
            self.assertIsInstance(e, Exception)

    def test_review_success(self):
        client = ClientReview()
        fit_config = {PrConfig.REVIEW_FLAG: True}
        _, _, train_conf = client.fit(self.parameters, fit_config)
        self.assertTrue(train_conf[PrConfig.REVIEW_FLAG])

    def test_review_failure(self):
        client = ClientReview()
        fit_config = {PrConfig.REVIEW_FLAG: False}
        try:
            client.fit(self.parameters, fit_config)
        except Exception as e:
            self.assertIsInstance(e, Exception)


if __name__ == "__main__":
    unittest.main()
