"""
Test basic properties of implemented datasets.
"""
import unittest

from classicdata import Ionosphere


class TestLoading(unittest.TestCase):
    def test_loading(self):
        ionosphere = Ionosphere()
        ionosphere.load()
        self.assertEqual(
            ionosphere.points.shape, (ionosphere.n_samples, ionosphere.n_features)
        )
        self.assertEqual(ionosphere.labels.shape, (ionosphere.n_samples,))


if __name__ == "__main__":
    unittest.main()
