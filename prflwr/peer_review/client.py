from abc import abstractmethod
from typing import Dict, List, Tuple

import numpy as np
from flwr.client import NumPyClient
from flwr.common import Scalar
from overrides import overrides
from prflwr.peer_review.config import PrConfig


class PeerReviewClient(NumPyClient):
    """Abstract class federated learning clients should extend implementing missing methods,
    which provides the routing of incoming packets from the server to the train and review methods.
    """

    @abstractmethod
    def review(
        self, parameters: List[np.ndarray], config: Dict[str, Scalar]
    ) -> Tuple[List[np.ndarray], int, Dict[str, Scalar]]:
        """Review the provided weights using the locally stored dataset.

        Parameters
        ----------
        parameters : List[np.ndarray]
            The current (global) model parameters.
        config : Dict[str, Scalar]
            Configuration parameters which allow the server to influence reviewing
            on the client. It can be used to communicate arbitrary values from the
            server to the client.

        Returns
        -------
        parameters : List[numpy.ndarray]
            The locally updated model parameters.
        num_examples : int
            The number of examples used for reviewing.
        metrics : Dict[str, Scalar]
            A dictionary mapping arbitrary string keys to values of type
            bool, bytes, float, int, or str. It can be used to communicate
            arbitrary values back to the server.
        """

    @abstractmethod
    def train(
        self, parameters: List[np.ndarray], config: Dict[str, Scalar]
    ) -> Tuple[List[np.ndarray], int, Dict[str, Scalar]]:
        """Train the provided parameters using the locally stored dataset.

        Parameters
        ----------
        parameters : List[numpy.ndarray]
            The current (global) model parameters.
        config : Dict[str, Scalar]
            Configuration parameters which allow the server to influence training
            on the client. It can be used to communicate arbitrary values from the
            server to the client.

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
            parameters, num_examples, metrics = self.review(parameters, config)
            metrics[PrConfig.REVIEW_FLAG] = True
        else:
            parameters, num_examples, metrics = self.train(parameters, config)
            metrics[PrConfig.REVIEW_FLAG] = False
        return parameters, num_examples, metrics
