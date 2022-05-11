import random
from collections import OrderedDict
from typing import List

import numpy as np
import torch


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


def set_parameters(model, parameters: List[np.ndarray]):
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
