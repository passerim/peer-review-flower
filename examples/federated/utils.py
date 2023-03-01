import flwr as fl
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from examples.centralized.centralized import Net, load_data
from examples.centralized.utils import set_seed


def client_fn(
    cid: str,
    num_clients: int,
    client_class: fl.client.NumPyClient,
    train_test_fraction: float = 1.0,
    data_path: str = "./data/cifar10",
    batch_size: int = 50,
    device: str = "cpu",
    seed: int = 0,
) -> fl.client.NumPyClient:
    set_seed(seed)

    # Load model
    net = Net().to(device)

    # Load data
    trainset, testset = load_data(data_path)
    trainset = Subset(
        trainset, list(range(int(train_test_fraction * trainset.data.shape[0])))
    )
    testset = Subset(
        testset, list(range(int(train_test_fraction * testset.data.shape[0])))
    )
    trainset_sampler = DistributedSampler(
        trainset, num_replicas=num_clients, rank=int(cid), shuffle=True, seed=seed
    )
    trainloader = DataLoader(trainset, sampler=trainset_sampler, batch_size=batch_size)
    testset_sampler = DistributedSampler(
        testset, num_replicas=num_clients, rank=int(cid), shuffle=True, seed=seed
    )
    testloader = DataLoader(testset, sampler=testset_sampler, batch_size=batch_size)
    return client_class(net, trainloader, testloader)
