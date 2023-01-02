import unittest

import numpy as np

from prflwr.utils import non_iid_partitions

CONCENTRATION = 1
NUM_CLASSES = 10
NUM_EXAMPLES = 60000
NUM_PARTITIONS = 50


class TestDataset(unittest.TestCase):
    def setUp(self) -> None:
        idxs = np.asarray(range(NUM_EXAMPLES))
        targets = np.random.randint((NUM_CLASSES,) * NUM_EXAMPLES)
        self.dataset = [idxs, targets]

    def test_correct_split(self):
        partitions = non_iid_partitions(self.dataset, NUM_PARTITIONS, CONCENTRATION)
        self.assertIsInstance(partitions, list)
        self.assertEqual(len(partitions), NUM_PARTITIONS)
        self.assertEqual(
            max([len(p) for p in partitions]), NUM_EXAMPLES // NUM_PARTITIONS
        )

    def test_less_samples_than_partitions(self):
        partitions = non_iid_partitions(
            (
                self.dataset[0][: NUM_PARTITIONS - 1],
                self.dataset[1][: NUM_PARTITIONS - 1],
            ),
            NUM_PARTITIONS,
            CONCENTRATION,
        )
        self.assertIsInstance(partitions, list)
        self.assertEqual(len(partitions), NUM_PARTITIONS)
        self.assertEqual(max([len(p) for p in partitions]), 1)
        self.assertEqual(len(partitions[-1]), 0)


if __name__ == "__main__":
    unittest.main()
