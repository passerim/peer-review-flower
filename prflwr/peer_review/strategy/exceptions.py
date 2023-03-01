class StrategyException(Exception):
    """A generic strategy exception."""

    pass


class ConfigureTrainException(StrategyException):
    """An exception that corresponds to an error in the
    `prflwr.peer_review.strategy.PeerReviewStrategy.configure_train` method."""

    pass


class ConfigureEvaluateException(StrategyException):
    """An exception that corresponds to an error in the
    `flwr.server.strategy.Strategy.configure_train` method."""

    pass


class ConfigureReviewException(StrategyException):
    """An exception that corresponds to an error in the
    `prflwr.peer_review.strategy.PeerReviewStrategy.configure_review`
    method."""

    pass


class AggregateTrainException(StrategyException):
    """An exception that corresponds to an error in the
    `prflwr.peer_review.strategy.PeerReviewStrategy.aggregate_train` method."""

    pass


class AggregateEvaluateException(StrategyException):
    """An exception that corresponds to an error in the
    `flwr.server.strategy.Strategy.aggregate_evaluate` method."""

    pass


class AggregateReviewException(StrategyException):
    """An exception that corresponds to an error in the
    `prflwr.peer_review.strategy.PeerReviewStrategy.aggregate_review`
    method."""

    pass


class AggregateAfterReviewException(StrategyException):
    """An exception that corresponds to an error in the
    `prflwr.peer_review.strategy.PeerReviewStrategy.aggregate_after_review`
    method."""

    pass


class StopReviewException(StrategyException):
    """An exception that corresponds to an error in the
    `prflwr.peer_review.strategy.PeerReviewStrategy.stop_review` method."""

    pass


class EvaluateException(StrategyException):
    """An exception that corresponds to an error in the
    `flwr.server.strategy.Strategy.evaluate` method."""

    pass


class InitializeParametersException(StrategyException):
    """An exception that corresponds to an error in the
    `flwr.server.strategy.Strategy.initialize_parameters` method."""

    pass
