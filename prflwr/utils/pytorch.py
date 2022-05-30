import random
from collections import OrderedDict
from typing import List

import numpy as np
import torch

import flwr as fl


def set_seed(seed):
    """Sets specified seed in Python random library, NumPy, PyTorch and Cuda.

    Parameters
    ----------
    seed : int
        Integer value of seed to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_parameters(model) -> List[np.ndarray]:
    """Returns a PyTorch model parameters as a list of NumPy multi-dimensional arrays.

    Parameters
    -------
    model : torch.nn.Module
        A PyTorch model.

    Returns
    -------
    List[np.ndarray]
        Model parameters as a list of NumPy multi-dimensional arrays.
    """
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: torch.nn.ModuleList, parameters: List[np.ndarray]):
    """Sets a list of NumPy multi-dimensional arrays as a PyTorch model parameters.

    Parameters
    -------
    model : torch.nn.Module
        A PyTorch model.
    List[np.ndarray]
        Model parameters as a list of NumPy multi-dimensional arrays.
    """
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def set_weights(model: torch.nn.Module, weights: fl.common.Weights) -> None:
    """Set model weights from a list of NumPy ndarrays."""
    state_dict = OrderedDict(
        {
            k: torch.tensor(np.atleast_1d(v))
            for k, v in zip(model.state_dict().keys(), weights)
        }
    )
    model.load_state_dict(state_dict, strict=True)
