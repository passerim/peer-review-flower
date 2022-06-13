import math
import multiprocessing as mp
import os
import re
import unittest

from examples.sim_peer_reviewed.simulation import setup_server

LOGGING_FILE = "./tests/test_sim_peer_reviewed.log"
TRAIN_TEST_FRACTION = 0.1
NUM_CLASSES = 10
NUM_CLIENTS = 2
FL_ROUNDS = 1
ctx = mp.get_context("spawn")


def run_fl():
    simulation = ctx.Process(
        target=setup_server, args=(FL_ROUNDS, NUM_CLIENTS, LOGGING_FILE)
    )
    simulation.start()
    simulation.join()


class TestPeerReviewedTraining(unittest.TestCase):
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
        self.setup_done = True
        if os.path.exists(LOGGING_FILE):
            os.remove(LOGGING_FILE)
        run_fl()

    def test_logging_file(self):
        self.assertTrue(os.path.exists(LOGGING_FILE))

    def test_pr_finished(self):
        with open(LOGGING_FILE, "r") as f:
            lines = f.readlines()
            self.assertGreater(sum([1 for line in lines if "FL finished" in line]), 0)

    def test_pr_loss(self):
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
