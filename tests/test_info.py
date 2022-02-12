"""
Test information.
"""

import unittest

from classicdata.info import list_datasets


class InfoTest(unittest.TestCase):
    def test_list(self):
        list_datasets(table_format="simple")
        list_datasets(table_format="github")


if __name__ == "__main__":
    unittest.main()
