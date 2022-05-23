import math
import os
import re
import unittest
from multiprocessing import Process

from examples.federated.client import setup_client
from examples.federated.server import setup_server

LOGGING_FILE = "./tests/test_federated.log"
NUM_CLASSES = 10
NUM_CLIENTS = 2
FL_ROUNDS = 1
PORT = 8089


def run_fl():
    server = Process(target=setup_server, args=(PORT, FL_ROUNDS, LOGGING_FILE))
    server.start()

    clients = list()
    for i in range(NUM_CLIENTS):
        c = Process(target=setup_client, args=(PORT, NUM_CLIENTS, i))
        c.start()
        clients.append(c)

    for c in clients:
        c.join()
    server.join()


class TestFederatedTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.setup_done = False
        cls.setUp(cls)

    @classmethod
    def tearDownClass(cls) -> None:
        if os.path.exists(LOGGING_FILE):
            os.remove(LOGGING_FILE)

    def setUp(self) -> None:
        if self.setup_done:
            return
        if os.path.exists(LOGGING_FILE):
            os.remove(LOGGING_FILE)
        run_fl()
        self.setup_done = True

    def test_logging_file(self):
        self.assertTrue(os.path.exists(LOGGING_FILE))

    def test_fl_finished(self):
        with open(LOGGING_FILE, "r") as f:
            lines = f.readlines()
            self.assertGreater(sum([1 for line in lines if "FL finished" in line]), 0)

    def test_fl_loss(self):
        with open(LOGGING_FILE, "r") as f:
            lines = f.readlines()
            for line in lines:
                if "losses_distributed" in line:
                    loss_str = re.search(r"\(.+?\)", line).group(0)
                    loss_str = (
                        loss_str.replace("(", "").replace(")", "").replace(" ", "")
                    )
                    loss = float(loss_str.split(",")[1])
                    self.assertLess(loss, -math.log(1 / NUM_CLASSES))
                    return
        self.assertTrue(False)


if __name__ == "__main__":
    unittest.main()
