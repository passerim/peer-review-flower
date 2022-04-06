from abc import abstractmethod
from typing import Dict, List, Tuple
from overrides import overrides

import numpy as np
from flwr.common import Scalar
from flwr.client import NumPyClient

from .prconfig import *


class PeerReviewClient(NumPyClient):

    @abstractmethod
    def review(
        self, parameters: List[np.ndarray], config: Dict[str, Scalar]
    ) -> Tuple[List[np.ndarray], int, Scalar]:
        """Review the provided weights using the locally held dataset.

        Parameters
        ----------
        parameters : List[np.ndarray]
            The current (global) model parameters.
        config : Dict[str, Scalar]
            Configuration parameters which allow the server to influence
            review on the client. It can be used to communicate
            arbitrary values from the server to the client, for example,
            to influence the number of examples used for reviewing.

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
       
    @abstractmethod
    def train(
        self, parameters: List[np.ndarray], config: Dict[str, Scalar]
    ) -> Tuple[List[np.ndarray], int, Dict[str, Scalar]]:
        """Train the provided parameters using the locally held dataset.

        Parameters
        ----------
        parameters : List[numpy.ndarray]
            The current (global) model parameters.
        config : Dict[str, Scalar]
            Configuration parameters which allow the
            server to influence training on the client. It can be used to
            communicate arbitrary values from the server to the client, for
            example, to set the number of (local) training epochs.

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
        is_review = config.get(REVIEW_FLAG)
        if is_review:
            parameters, num_examples, loss = self.review(parameters, config)
            return parameters, num_examples, {
                REVIEW_FLAG: True,
                REVIEW_SCORE: float(loss),
            }
        else:
            parameters, num_examples, metrics = self.train(parameters, config)
            metrics[REVIEW_FLAG] = False
            return parameters, num_examples, metrics
