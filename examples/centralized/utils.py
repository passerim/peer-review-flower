import random
from collections import OrderedDict

import numpy as np
import torch
from flwr.common import NDArrays


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


def get_parameters(model) -> NDArrays:
    """Returns a PyTorch model parameters as a list of numpy.ndarray.

    Parameters
    -------
    model : torch.nn.Module
        A PyTorch model.

    Returns
    -------
    NDArrays
        Model parameters as a list of numpy.ndarray objects.
    """
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: torch.nn.Module, parameters: NDArrays):
    """Sets a list of numpy.ndarray objects as a PyTorch model parameters.

    Parameters
    -------
    model : torch.nn.Module
        A PyTorch model.
    parameters: NDArrays
        Model parameters as a list of numpy.ndarray objects.
    """
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def set_weights(model: torch.nn.Module, weights: NDArrays) -> None:
    """Set model weights from a list of numpy.ndarray objects."""
    state_dict = OrderedDict(
        {
            k: torch.tensor(np.atleast_1d(v))
            for k, v in zip(model.state_dict().keys(), weights)
        }
    )
    model.load_state_dict(state_dict, strict=True)
