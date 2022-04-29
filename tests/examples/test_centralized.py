import math
import unittest

import torch
from prflwr.utils.pytorch import set_seed
from torch.utils.data import DataLoader

from examples.centralized.centralized import Net, load_data, test, train

SEED = 0
BATCH_SIZE = 32
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TestCentralizedTraining(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.setup_done = False
        cls.setUp(cls)

    def setUp(self) -> None:
        if self.setup_done:
            return

        # Load model
        set_seed(SEED)
        net = Net().to(DEVICE)

        # Load data
        trainset, testset, _ = load_data()
        self.num_classes = len(trainset.classes)
        trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
        testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

        # Check metrics before training
        self.loss_start, self.accuracy_start = test(net, testloader, device=DEVICE)

        # Start centralized training
        train(net, trainloader, epochs=1, device=DEVICE, verbose=True)

        # Check metrics after training
        self.loss_after_epoch, self.accuracy_after_epoch = test(net, testloader, device=DEVICE)
        self.setup_done = True

    def test_loss_start(self):
        self.assertAlmostEqual(self.loss_start, -math.log(1/self.num_classes), 1)

    def test_accuracy_start(self):
        self.assertAlmostEqual(self.accuracy_start, 1/self.num_classes, 1)

    def test_loss_after_epoch(self):
        self.assertLess(self.loss_after_epoch, -math.log(1/self.num_classes))

    def test_accuracy_after_epoch(self):
        self.assertGreater(self.accuracy_after_epoch, 1/self.num_classes)


if __name__ == "__main__":
    unittest.main()
