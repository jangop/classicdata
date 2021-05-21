"""
Base classes for datasets.
"""
from abc import abstractmethod
from typing import Optional

import numpy as np
from sklearn.preprocessing import LabelEncoder


class Dataset:
    """
    Abstract base class for datasets.
    """

    def __init__(self, safe_name: str, short_name: str, long_name: str):
        self.safe_name = safe_name
        self.short_name = short_name
        self.long_name = long_name

        self.points: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.label_encoder = LabelEncoder()
        self.loaded: bool = False

    @abstractmethod
    def load(self):
        """
        Expected to fill self.points and self.labels, fit self.label_encoder, and set self.loaded.
        """
        pass
