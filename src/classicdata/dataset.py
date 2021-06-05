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

    # pylint: disable=too-many-instance-attributes
    # This class is supposed to provide plenty of information in a convenient way.

    def __init__(
        self,
        *,
        safe_name: str,
        short_name: str,
        long_name: str,
        n_samples: int,
        n_features: int,
        n_classes: int,
    ):
        self.safe_name = safe_name
        self.short_name = short_name
        self.long_name = long_name
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_classes = n_classes

        self._points: Optional[np.ndarray] = None
        self._targets: Optional[np.ndarray] = None
        self.label_encoder = LabelEncoder()
        self.loaded: bool = False

    @property
    def points(self) -> np.ndarray:
        """
        All points of this dataset.
        """
        if not self.loaded:
            self.load()
        return self._points

    @property
    def targets(self) -> np.ndarray:
        """
        All targets of this dataset, matching `.points`.
        """
        if not self.loaded:
            self.load()
        return self._targets

    @property
    def labels(self) -> np.ndarray:
        """
        Alias of `.targets`.
        """
        return self.targets

    @abstractmethod
    def load(self):
        """
        Expected to fill self._points and self._targets,
        fit self.label_encoder, and set self.loaded.
        """

    def decode_labels(self, encoded_labels):
        """
        Decode encoded labels.
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
            train_index = np.ones_like(self.targets, dtype=bool)
            test_index = np.zeros_like(self.targets, dtype=bool)
        else:
            splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=test_size,
                train_size=train_size,
                random_state=DEFAULT_RANDOM_STATE,
            )
            train_index, test_index = next(splitter.split(self.points, self.targets))

        scaler = StandardScaler()
        train_points = scaler.fit_transform(self.points[train_index])
        train_targets = self.targets[train_index]
        if test_size == 0:
            test_points = self.points[test_index]
        else:
            test_points = scaler.transform(self.points[test_index])
        test_targets = self.targets[test_index]

        return train_points, train_targets, test_points, test_targets
