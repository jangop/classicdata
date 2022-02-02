"""
Miscellaneous
"""

import numpy as np

from classicdata.dataset import Source


def translate(array: np.ndarray, table: dict):
    """
    Replace all objects in an array according to a table.
    """
    return np.ascontiguousarray([table[key] for key in array])


uci_ml_repo = Source(
    name="UCI Machine Learning Repository",
    url="https://archive.ics.uci.edu",
    citation_url="https://archive.ics.uci.edu/ml/citation_policy.html",
)
