import warnings

import torch
from prflwr.utils.pytorch import set_seed
from torch.utils.data import DataLoader

from .utils import load_data, load_efficientnet, test, train

warnings.filterwarnings("ignore")

SEED = 0
BATCH_SIZE = 32
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    set_seed(SEED)

    # Load model
    net = load_efficientnet(finetune=False, classes=100)

    # Load data
    trainset, testset, num_examples = load_data()
    print("Data samples:", num_examples)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)

    # Start centralized training
    train(net, trainloader, epochs=1, device=DEVICE)
    loss, accuracy = test(net, testloader, device=DEVICE)
    print(f"Final test set performance: loss {loss}, accuracy {accuracy}")


if __name__ == "__main__":
    main()
