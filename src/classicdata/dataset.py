"""
Base classes for datasets.
"""
from abc import abstractmethod
from typing import Optional, Union

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from classicdata.settings import DEFAULT_TEST_SIZE, DEFAULT_RANDOM_STATE


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

    def decode_labels(self, encoded_labels):
        """
        Decode labels.
        """
        return self.label_encoder.inverse_transform(encoded_labels)

    def split_for_training(
        self,
        test_size: Optional[Union[float, int]] = None,
        train_size: Optional[Union[float, int]] = None,
    ):
        """
        Split and scale the dataset.
        """
        if test_size is None:
            test_size = DEFAULT_TEST_SIZE

        if test_size == 0:
            train_index = np.ones_like(self.labels, dtype=bool)
            test_index = np.zeros_like(self.labels, dtype=bool)
        else:
            splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=test_size,
                train_size=train_size,
                random_state=DEFAULT_RANDOM_STATE,
            )
            train_index, test_index = next(splitter.split(self.points, self.labels))

        scaler = StandardScaler()
        train_points = scaler.fit_transform(self.points[train_index])
        train_labels = self.labels[train_index]
        if test_size == 0:
            test_points = self.points[test_index]
        else:
            test_points = scaler.transform(self.points[test_index])
        test_labels = self.labels[test_index]

        return train_points, train_labels, test_points, test_labels
