"""
Test file handling.
"""

import unittest

from classicdata.files import provide_file


class HashTest(unittest.TestCase):
    def test_invalid_hash(self):
        with self.assertRaises(RuntimeError):
            provide_file(
                url="https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data",
                expected_hash="foobar",
            )


if __name__ == "__main__":
    unittest.main()
