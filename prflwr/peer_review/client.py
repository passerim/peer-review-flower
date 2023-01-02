from abc import abstractmethod

from flwr.client import Client
from flwr.common import FitIns, FitRes
from overrides import overrides

from prflwr.peer_review.config import PrConfig
from prflwr.peer_review.typing import ReviewIns, ReviewRes, TrainIns, TrainRes


class PeerReviewClient(Client):
    """Abstract class that peer review clients should extend by implementing
    missing methods, which provides the routing of incoming packets from the
    server to the train and review methods."""

    @abstractmethod
    def train(self, ins: TrainIns) -> TrainRes:
        """Refine the provided parameters using the locally held dataset.
        Parameters
        ----------
        ins : TrainIns
            The training instructions containing (global) model parameters
            received from the server and a dictionary of configuration values
            used to customize the local training process.
        Returns
        -------
        TrainRes
            The training result containing updated parameters and other details
            such as the number of local training examples used for training.
        """

    @abstractmethod
    def review(self, ins: ReviewIns) -> ReviewRes:
        """Refine the provided parameters using the locally held dataset.
        Parameters
        ----------
        ins : ReviewIns
            The review instructions containing (global) candidate parameters
            received from the server and a dictionary of configuration values
            used to customize the local review process.
        Returns
        -------
        ReviewRes
            The review result containing the parameters and other details
            such as the number of local training examples used for reviewing.
        """

    @overrides
    def fit(self, ins: FitIns) -> FitRes:
        config = ins.config
        is_review = config.get(PrConfig.REVIEW_FLAG)
        if is_review:
            config.pop(PrConfig.REVIEW_FLAG, None)
            ins = ReviewIns(ins.parameters, config)
            res = self.review(ins)
            res.metrics[PrConfig.REVIEW_FLAG] = True
        else:
            config.pop(PrConfig.REVIEW_FLAG, None)
            ins = TrainIns(ins.parameters, config)
            res = self.train(ins)
            res.metrics[PrConfig.REVIEW_FLAG] = False
        return FitRes(res.status, res.parameters, res.num_examples, res.metrics)
