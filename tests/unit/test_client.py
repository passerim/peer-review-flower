import unittest

import numpy as np
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    Status,
    ndarrays_to_parameters,
)
from overrides import overrides

from prflwr.peer_review import (
    PeerReviewClient,
    PrConfig,
    ReviewIns,
    ReviewRes,
    TrainIns,
    TrainRes,
)


class FailingClient(PeerReviewClient):
    """A client which always fails - raises Exception or NotImplementedError upon any method call."""

    @overrides
    def get_properties(self, ins: GetPropertiesIns) -> GetPropertiesRes:
        raise NotImplementedError

    @overrides
    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        raise NotImplementedError

    @overrides
    def train(self, ins: TrainIns) -> TrainRes:
        raise Exception

    @overrides
    def review(self, ins: ReviewIns) -> ReviewRes:
        raise Exception

    @overrides
    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        raise Exception


class ClientTrain(FailingClient):
    """A client with a train method implementation."""

    @overrides
    def train(self, ins: TrainIns) -> TrainRes:
        return TrainRes(Status(Code.OK, ""), ins.parameters, 0, {})


class ClientReview(FailingClient):
    """A client with a review method implementation."""

    @overrides
    def review(self, ins: ReviewIns) -> ReviewRes:
        return ReviewRes(
            Status(Code.OK, ""), ins.parameters, 0, {PrConfig.REVIEW_SCORE: 1}
        )


class TestPeerReviewClient(unittest.TestCase):
    """Tests correct routing of PeerReviewNumPyClient fit method calls to train
    and review methods of subclasses based on flag value."""

    parameters = ndarrays_to_parameters([np.array([[1, 2], [3, 4], [5, 6]])])

    def test_train_success(self):
        client = ClientTrain()
        fit_config = {PrConfig.REVIEW_FLAG: False}
        ins = TrainIns(self.parameters, fit_config)
        res = client.fit(ins)
        self.assertFalse(res.metrics[PrConfig.REVIEW_FLAG])

    def test_train_failure(self):
        client = ClientTrain()
        fit_config = {PrConfig.REVIEW_FLAG: False}
        ins = TrainIns(self.parameters, fit_config)
        try:
            client.fit(ins)
        except Exception as e:
            self.assertIsInstance(e, Exception)

    def test_review_success(self):
        client = ClientReview()
        fit_config = {PrConfig.REVIEW_FLAG: True}
        ins = ReviewIns(self.parameters, fit_config)
        res = client.fit(ins)
        self.assertTrue(res.metrics[PrConfig.REVIEW_FLAG])

    def test_review_failure(self):
        client = ClientReview()
        fit_config = {PrConfig.REVIEW_FLAG: False}
        ins = ReviewIns(self.parameters, fit_config)
        try:
            client.fit(ins)
        except Exception as e:
            self.assertIsInstance(e, Exception)


if __name__ == "__main__":
    unittest.main()
