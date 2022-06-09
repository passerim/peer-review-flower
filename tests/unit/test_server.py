import unittest
from unittest.mock import Mock

from flwr.server.client_manager import SimpleClientManager
from flwr.server.strategy.fedavg import FedAvg
from prflwr.peer_reviewed.server import PeerReviewServer
from prflwr.peer_reviewed.strategy.strategy import PeerReviewStrategy

TEST_VALUE = 42


class EmptyStrategy(PeerReviewStrategy):
    pass


class FailingStrategy(PeerReviewStrategy):
    pass


class SuccessfulStrategy(PeerReviewStrategy):
    pass


class TestPeerReviewServer(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
