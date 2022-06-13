import unittest

from prflwr.utils import import_dataset_utils


class TestFlowerUtils(unittest.TestCase):
    def test_dataset_import(self):
        import_dataset_utils()
        from flwr.dataset.utils import common

        self.assertIsNotNone(common)


if __name__ == "__main__":
    unittest.main()
