import argparse
from collections import OrderedDict

import flwr as fl
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from ..centralized.centralized import Net, load_data, train, test
from ..utils.utils import set_seed
from .prclient import PeerReviewClient


SEED = 0
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32


class CifarClient(PeerReviewClient):

    def __init__(self, model: torch.nn.Module, trainloader, testloader, num_examples):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.num_examples = num_examples

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def train(self, parameters, config):
        self.set_parameters(parameters)
        train(self.model, self.trainloader, epochs=1)
        return self.get_parameters(), self.num_examples["trainset"], {}

    def review(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.testloader)
        return self.get_parameters(), self.num_examples["trainset"], float(loss)

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.testloader)
        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}


def setup_client(port: int, num_clients: int, partition: int):
    
    set_seed(SEED)

    # Load model
    net = Net().to(DEVICE)

    # Load data
    trainset, testset, num_examples = load_data()
    trainset_sampler = DistributedSampler(trainset, num_replicas=num_clients, rank=partition, shuffle=True, seed=SEED)
    trainloader = DataLoader(trainset, shuffle=False, sampler=trainset_sampler, batch_size=BATCH_SIZE)
    testset_sampler = DistributedSampler(testset, num_replicas=num_clients, rank=partition, shuffle=True, seed=SEED)
    testloader = DataLoader(testset, shuffle=False, sampler=testset_sampler, batch_size=BATCH_SIZE)

    # Start client
    fl.client.start_numpy_client(
        f"localhost:{port}", client=CifarClient(
            net, trainloader, testloader, num_examples
        )
    )


def main():
    """Create model, load data, define Flower client, start Flower client."""

    # Parse command line argument `partition`
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, choices=range(0, 65535), required=True)
    parser.add_argument("--num_clients", type=int, choices=range(1, 1000), required=True)
    parser.add_argument("--partition", type=int, choices=range(0, 1000), required=True)
    args = parser.parse_args()

    if args.partition >= args.num_clients:
        print("The selected partition of the training data should be an integer lower than the number of clients.")
        exit()
    
    setup_client(args.port, args.num_clients, args.partition)
    

if __name__ == "__main__":
    main()
