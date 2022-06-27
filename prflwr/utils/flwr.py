import importlib


def import_dataset_utils():
    """Import Flower's package for performing various pre-processing operations
    on datasets, in order to use them in a federated learning setting.
    """
    importlib.import_module(".common", "flwr.dataset.utils")
