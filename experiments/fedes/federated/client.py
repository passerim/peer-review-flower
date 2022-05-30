import flwr as fl
from prflwr.utils.pytorch import get_parameters, set_parameters
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from ..resnet import WideResNet
from ..utils import train


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

    def set_parameters(self, parameters, model_config):
        """Loads WideResNet model and replaces its parameters with the given ones."""
        model_opt = model_config["model_opt"]
        model = WideResNet(
            model_opt["batch_size"], model_opt["batch_size"], model_opt["num_classes"]
        )
        set_parameters(model, parameters)
        return model

    def fit(self, parameters, config):
        """Trains parameters on the locally held training set."""
        model = self.set_parameters(parameters, config["model_opt"])
        trainLoader = DataLoader(
            self.trainset, batch_size=config["batch_size"], shuffle=True
        )
        criterion = nn.CrossEntropyLoss().to(self.device)
        optimizer = optim.SGD(
            model.parameters(), lr=config["lr"], momentum=0.9, weight_decay=5e-04
        )
        for epoch in range(1, config["local_epochs"] + 1):
            loss, accuracy, time = train(
                model, trainLoader, criterion, optimizer, self.device, display=False
            )
        parameters_prime = get_parameters(model)
        num_examples_train = len(self.trainset)
        return parameters_prime, num_examples_train, dict()

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""
        raise NotImplementedError
