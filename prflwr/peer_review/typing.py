from dataclasses import dataclass

from flwr.common import FitIns, FitRes

from prflwr.peer_review.config import PrConfig


class TrainIns(FitIns):
    """Train instructions for a client."""

    def __post_init__(self):
        self.config[PrConfig.REVIEW_FLAG] = False

    def __setattr__(self, attr, value):
        super().__setattr__(attr, value)
        if attr == "config":
            self.config[PrConfig.REVIEW_FLAG] = False


@dataclass(init=False)
class ReviewIns(FitIns):
    """Review instructions for a client."""

    def __post_init__(self):
        self.config[PrConfig.REVIEW_FLAG] = True

    def __setattr__(self, attr, value):
        super().__setattr__(attr, value)
        if attr == "config":
            self.config[PrConfig.REVIEW_FLAG] = True


@dataclass(init=False)
class TrainRes(FitRes):
    """Train response from a client."""

    def __post_init__(self):
        self.config[PrConfig.REVIEW_FLAG] = False

    def __setattr__(self, attr, value):
        super().__setattr__(attr, value)
        if attr == "config":
            self.config[PrConfig.REVIEW_FLAG] = False


@dataclass(init=False)
class ReviewRes(FitRes):
    """Review response from a client."""

    def __post_init__(self):
        self.config[PrConfig.REVIEW_FLAG] = True

    def __setattr__(self, attr, value):
        super().__setattr__(attr, value)
        if attr == "config":
            self.config[PrConfig.REVIEW_FLAG] = True
