import unittest

from src.stats import mean


class MeanTest(unittest.TestCase):
    def test_mean_of_three_values(self):
        self.assertEqual(mean([2, 4, 6]), 4)


if __name__ == "__main__":
    unittest.main()
