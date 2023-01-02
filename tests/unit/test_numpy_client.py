import unittest
from typing import Dict, Tuple

import numpy as np
from flwr.common import Config, NDArrays, Properties, Scalar
from overrides import overrides

from prflwr.peer_review import PeerReviewNumPyClient, PrConfig


class FailingClient(PeerReviewNumPyClient):
    """A client which always fails - raises Exception or NotImplementedError upon any method call."""

    @overrides
    def get_properties(self, config: Config) -> Properties:
        raise NotImplementedError

    @overrides
    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        raise NotImplementedError

    @overrides
    def train(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        raise Exception

    @overrides
    def review(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        raise Exception

    @overrides
    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        raise Exception


class ClientTrain(FailingClient):
    """A client with a train method implementation."""

    @overrides
    def train(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        return parameters, 0, {}


class ClientReview(FailingClient):
    """A client with a review method implementation."""

    @overrides
    def review(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        return parameters, 0, {PrConfig.REVIEW_SCORE: 1}


class TestPeerReviewClient(unittest.TestCase):
    """Tests correct routing of PeerReviewNumPyClient fit method calls to train
    and review methods of subclasses based on flag value."""

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
