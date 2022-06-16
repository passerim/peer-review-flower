class StrategyException(Exception):
    pass


class ConfigureTrainException(StrategyException):
    pass


class ConfigureEvaluateException(StrategyException):
    pass


class ConfigureReviewException(StrategyException):
    pass


class AggregateTrainException(StrategyException):
    pass


class AggregateEvaluateException(StrategyException):
    pass


class AggregateReviewException(StrategyException):
    pass


class AggregateAfterReviewException(StrategyException):
    pass


class StopReviewException(StrategyException):
    pass


class EvaluateException(StrategyException):
    pass


class InitializeParametersException(StrategyException):
    pass
