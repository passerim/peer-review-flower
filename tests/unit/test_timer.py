import time
import unittest

from prflwr.utils import FitTimer


class TestFitTimer(unittest.TestCase):
    def setUp(self) -> None:
        self.timer = FitTimer()

    def test_init(self):
        self.assertEqual(0.0, self.timer.get_elapsed())
        self.assertFalse(self.timer.is_on())

    def test_stop(self):
        self.timer.start()
        time.sleep(1)
        self.timer.stop()
        elapsed = self.timer.get_elapsed()
        time.sleep(1)
        self.assertEqual(elapsed, self.timer.get_elapsed())

    def test_start(self):
        self.timer.start()
        time.sleep(1)
        elapsed = self.timer.get_elapsed()
        self.assertGreater(elapsed, 0)

    def test_reset(self):
        self.timer.start()
        self.timer.reset(5)
        self.assertFalse(self.timer.is_on())
        self.assertEqual(self.timer.get_elapsed(), 5)

    def test_get_elapsed(self):
        self.assertEqual(self.timer.get_elapsed(), 0)
        self.timer.start()
        time.sleep(1)
        self.assertGreater(self.timer.get_elapsed(), 0)
        self.assertTrue(self.timer.is_on())

    def test_is_on(self):
        self.assertFalse(self.timer.is_on())
        self.timer.start()
        self.assertTrue(self.timer.is_on())
        self.timer.stop()
        self.assertFalse(self.timer.is_on())
        self.timer.start()
        self.timer.reset()
        self.assertFalse(self.timer.is_on())


if __name__ == "__main__":
    unittest.main()
