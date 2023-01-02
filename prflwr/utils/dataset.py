from typing import Generator, List, Optional, Tuple, Union

import numpy as np
from numpy.random.bit_generator import BitGenerator, SeedSequence

Dataset = Tuple[np.ndarray, np.ndarray]


def non_iid_partitions(
    dataset: Dataset,
    num_partitions: int,
    concentration: float,
    seed: Optional[Union[int, SeedSequence, BitGenerator, Generator]] = None,
) -> List[np.ndarray]:
    """Partition a dataset to produce non iid partitions sampling from a
    Dirichlet distribution.

    Parameters
    ----------
    dataset : Dataset
        A dataset given as a tuple of two numpy.ndarray, one holds the
        indices of the samples the other holds the sample's class/target.
    num_partitions : int
        The number of partitions the input dataset should be split into.
    concentration : float
        A single concentration value of the Dirichlet distribution to be sampled from.
    seed : Optional[Union[int, SeedSequence, BitGenerator, Generator]]
        From the official Numpy's documentation, it is defined as follows:
        "A seed to initialize the BitGenerator.
        [...]
        If an int or array_like[ints] is passed, then it will be passed
        to SeedSequence to derive the initial BitGenerator state.
        One may also pass in a SeedSequence instance.
        Additionally, when passed a BitGenerator, it will be wrapped by Generator.
        If passed a Generator, it will be returned unaltered."
        If None, then the legacy RandomState will be used for sampling.

    Returns
    -------
    partitions : List[numpy.ndarray]
        A List of as many numpy.ndarray as the required number of partitions,
        each of them holds the indices of the samples in the respective partion.
    """
    idxs, targets = dataset
    num_samples = len(idxs)
    if seed:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.RandomState(np.random.get_state()[1])
    classes, counts = np.unique(targets, return_counts=True)
    counts = counts / counts.sum()
    label_distribution = rng.dirichlet(
        [
            concentration,
        ]
        * len(classes),
        num_partitions,
    )
    label_distribution = label_distribution / counts
    label_distribution = label_distribution / label_distribution.sum(0)
    samples_in_partition = np.bincount(
        list(map(lambda x: x % num_partitions, range(num_samples))),
        minlength=num_partitions,
    )
    partition_idxs = list(range(num_partitions))
    partitions = [[] for _ in range(num_partitions)]
    for idx, target in zip(idxs, targets):
        p = rng.choice(partition_idxs, p=label_distribution[:, target])
        if len(partitions[p]) >= samples_in_partition[p]:
            while True:
                p = p + 1
                k = p % len(partitions)
                if len(partitions[k]) < samples_in_partition[k]:
                    partitions[k].append(idx)
                    break
        else:
            partitions[p].append(idx)
    return [np.asarray(partition) for partition in partitions]
