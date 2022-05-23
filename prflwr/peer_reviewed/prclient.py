from abc import abstractmethod
from typing import Dict, List, Tuple

import numpy as np
from flwr.client import NumPyClient
from flwr.common import Scalar
from overrides import overrides

from .prconfig import PrConfig


class PeerReviewClient(NumPyClient):
    """Abstract class clients should extend implementing missing methods, which provides
    the routing of incoming packets from the server to the train or review methods.
    """

    @abstractmethod
    def review(
        self, parameters: List[np.ndarray], config: Dict[str, Scalar]
    ) -> Tuple[List[np.ndarray], int, Scalar]:
        """Review the provided weights using the locally stored dataset.

        Parameters
        ----------
        parameters : List[np.ndarray]
            The current (global) model parameters sent by the centralized aggregator.
        config : Dict[str, Scalar]
            Configuration parameters which allow the aggregator to influence review
            on the client. It can be used to communicate arbitrary values from the
            aggregator to the client, for example, to influence the number of examples
            used for reviewing.

        Returns
        -------
        parameters : List[numpy.ndarray]
            The locally updated model parameters.
        num_examples : int
            The number of examples used for reviewing.
        metrics : Dict[str, Scalar]
            A dictionary mapping arbitrary string keys to values of type
            bool, bytes, float, int, or str. It can be used to communicate
            arbitrary values back to the aggregator.
        """

    @abstractmethod
    def train(
        self, parameters: List[np.ndarray], config: Dict[str, Scalar]
    ) -> Tuple[List[np.ndarray], int, Dict[str, Scalar]]:
        """Train the provided parameters using the locally stored dataset.

        Parameters
        ----------
        parameters : List[numpy.ndarray]
            The current (global) model parameters sent by the centralized aggregator.
        config : Dict[str, Scalar]
            Configuration parameters which allow the aggregator to influence training
            on the client. It can be used to communicate arbitrary values from the
            aggregator to the client, for example, to set the number of (local)
            training epochs.

        Returns
        -------
        parameters : List[numpy.ndarray]
            The locally updated model parameters.
        num_examples : int
            The number of examples used for training.
        metrics : Dict[str, Scalar]
            A dictionary mapping arbitrary string keys to values of type
            bool, bytes, float, int, or str. It can be used to communicate
            arbitrary values back to the server.
        """

    @overrides
    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, Scalar]
    ) -> Tuple[List[np.ndarray], int, Dict[str, Scalar]]:
        is_review = config.get(PrConfig.REVIEW_FLAG)
        if is_review:
            parameters, num_examples, loss = self.review(parameters, config)
            return (
                parameters,
                num_examples,
                {
                    PrConfig.REVIEW_FLAG: True,
                    PrConfig.REVIEW_SCORE: float(loss),
                },
            )
        else:
            parameters, num_examples, metrics = self.train(parameters, config)
            metrics[PrConfig.REVIEW_FLAG] = False
            return parameters, num_examples, metrics
