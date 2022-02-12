"""
Test file handling.
"""

import unittest

from classicdata.files import provide_file
from classicdata.settings import base_directory


class HashTest(unittest.TestCase):
    def test_invalid_hash(self):
        with self.assertRaises(RuntimeError):
            provide_file(
                url="https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data",
                root_directory=base_directory,
                sub_directory="foobar",
                expected_hash="foobar",
            )


if __name__ == "__main__":
    unittest.main()
