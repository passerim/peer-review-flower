import importlib


def import_dataset_utils():
    importlib.import_module(".common", "flwr.dataset.utils")
