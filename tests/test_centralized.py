import math
import unittest

import torch
from torch.utils.data import DataLoader

from src.centralized.centralized import load_data, train, test, Net
from src.utils.utils import set_seed

class TestCentralizedTraining(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.setup_done = False
        cls.setUp(cls)

    def setUp(self) -> None:

        if self.setup_done:
            return

        seed = 0
        batch_size = 32
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        set_seed(seed)

        # Load model
        net = Net().to(device)

        # Load data
        trainset, testset, _ = load_data()
        self.num_classes = len(trainset.classes)
        generator = torch.Generator()
        generator.manual_seed(seed)
        trainloader = DataLoader(trainset, batch_size=batch_size, 
                                 shuffle=True, generator=generator)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

        self.loss_start, self.accuracy_start = test(net, testloader)

        # Start centralized training
        train(net, trainloader, epochs=1, verbose=True)

        self.loss_after_epoch, self.accuracy_after_epoch = test(net, testloader)

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
