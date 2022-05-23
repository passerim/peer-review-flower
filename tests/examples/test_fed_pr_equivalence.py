import os
import re
import unittest

from .test_federated import run_fl as tf_run
from .test_peer_reviewed import run_fl as pr_run

LOGGING_FILES = ["./tests/test_federated.log", "./tests/test_peer_reviewed.log"]
NUM_CLASSES = 10
NUM_CLIENTS = 2
FL_ROUNDS = 1


class TestEquivTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.setup_done = False
        cls.setUp(cls)

    @classmethod
    def tearDownClass(cls) -> None:
        for file in LOGGING_FILES:
            if os.path.exists(file):
                os.remove(file)

    def setUp(self) -> None:
        if self.setup_done:
            return
        tf_run()
        pr_run()
        self.setup_done = True

    def test_logging_file(self):
        self.assertTrue(
            os.path.exists(LOGGING_FILES[0]) and os.path.exists(LOGGING_FILES[1])
        )

    def test_fl_finished(self):
        with open(LOGGING_FILES[0], "r") as f:
            lines = f.readlines()
            self.assertGreater(sum([1 for line in lines if "FL finished" in line]), 0)

    def test_pr_finished(self):
        with open(LOGGING_FILES[1], "r") as f:
            lines = f.readlines()
            self.assertGreater(sum([1 for line in lines if "FL finished" in line]), 0)

    def test_pr_loss(self):
        with open(LOGGING_FILES[0], "r") as f:
            lines = f.readlines()
            for line in lines:
                if "losses_distributed" in line:
                    loss_str = re.search(r"\(.+?\)", line).group(0)
                    loss_str = (
                        loss_str.replace("(", "").replace(")", "").replace(" ", "")
                    )
                    first_loss = float(loss_str.split(",")[1])
        with open(LOGGING_FILES[1], "r") as f:
            lines = f.readlines()
            for line in lines:
                if "losses_distributed" in line:
                    loss_str = re.search(r"\(.+?\)", line).group(0)
                    loss_str = (
                        loss_str.replace("(", "").replace(")", "").replace(" ", "")
                    )
                    second_loss = float(loss_str.split(",")[1])
        self.assertAlmostEqual(first_loss, second_loss, 1)


if __name__ == "__main__":
    unittest.main()
