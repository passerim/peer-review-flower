import flwr as fl
from prflwr.utils.pytorch import get_parameters, set_parameters
from torch.utils.data import DataLoader, Dataset

from ..utils import train, test, load_efficientnet

SEED = 0


class CifarClient(fl.client.NumPyClient):
    def __init__(
        self,
        trainset: Dataset,
        testset: Dataset,
        device: str = "cpu",
        cid: int = None,
    ):
        self.trainset = trainset
        self.testset = testset
        self.device = device
        self.cid = cid

    def get_parameters(self):
        """Get parameters of the local model."""
        return get_parameters(self.model)

    def set_parameters(self, parameters):
        """Loads an efficientnet model and replaces its parameters with the given ones."""
        model = load_efficientnet(finetune=False, classes=100)
        set_parameters(model, parameters)
        return model

    def fit(self, parameters, config):
        """Trains parameters on the locally held training set."""
        model = self.set_parameters(parameters)
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]
        trainLoader = DataLoader(self.trainset, batch_size=batch_size, shuffle=True)
        train(model, trainLoader, epochs, self.device)
        parameters_prime = get_parameters(model)
        num_examples_train = len(self.trainset)
        return parameters_prime, num_examples_train, dict()

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""
        model = self.set_parameters(parameters)
        testloader = DataLoader(self.testset, batch_size=32)
        loss, accuracy = test(model, testloader, self.device)
        return float(loss), len(self.testset), {"accuracy": float(accuracy)}
