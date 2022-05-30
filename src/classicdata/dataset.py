"""
Base classes for datasets.
"""
import enum
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .settings import DEFAULT_RANDOM_STATE, DEFAULT_TEST_SIZE


class CitationWarning(UserWarning):
    """
    Reminder to cite a dataset's source.
    """


class FeatureTypeMismatch(Exception):
    """
    Raised when a feature is of the wrong type.
    """


class FeatureTypeWarning(UserWarning):
    """
    Raised when a feature is likely of the wrong type.
    """


class FeatureType(enum.Enum):
    BINARY = enum.auto()
    CATEGORICAL = enum.auto()
    ORDINAL = enum.auto()
    NUMERICAL = enum.auto()
    MIXED = enum.auto()
    UNKNOWN = enum.auto()

    def matches(self, data: np.ndarray) -> bool:
        """
        Whether the data matches the feature type.
        """
        values, counts = np.unique(data, return_counts=True)
        n_values = len(values)
        n_points = len(data)
        if self == FeatureType.BINARY:
            if n_values > 2:
                return False
        elif self in (FeatureType.CATEGORICAL, FeatureType.ORDINAL):
            if n_values / n_points > 0.8:
                warnings.warn(
                    f"{self.name} feature has {n_values} unique values, "
                    f"which is over 80% of the total number of points ({n_points}).",
                    FeatureTypeWarning,
                )
        elif self == FeatureType.NUMERICAL:
            if n_values / n_points < 0.8:
                warnings.warn(
                    f"{self.name} feature has {n_values} unique values, "
                    f"which is less than 80% of the total number of points ({n_points}).",
                    FeatureTypeWarning,
                )
        return True


@dataclass
class Feature:
    short_name: str
    long_name: str
    type: FeatureType
    description: Optional[str] = None


@dataclass
class Source:
    """
    Contains information about where datasets come from.
    """

    name: str
    url: str
    citation_url: str

    mentioned: bool = field(default=False, repr=False, hash=False, compare=False)

    def mention(self):
        """
        Issue a reminder to cite the source.
        """
        if not self.mentioned:
            warnings.warn(
                f"You are using data from {self.name}. "
                f"When publishing your work, please consider appropriate citations. "
                f"See {self.citation_url}",
                CitationWarning,
            )
        self.mentioned = True


class Dataset(ABC):
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
        features: Optional[Sequence[Feature]],
        n_samples: Optional[int],
        n_features: Optional[int],
        n_classes: Optional[int],
        source: Optional[Source],
    ):
        self.safe_name = safe_name
        self.short_name = short_name
        self.long_name = long_name
        self.features = features
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_classes = n_classes
        self.source = source

        self._points: Optional[np.ndarray] = None
        self._targets: Optional[np.ndarray] = None
        self.label_encoder = LabelEncoder()
        self.loaded: bool = False

        if self.features is not None and self.n_features is not None:
            if len(self.features) != self.n_features:
                raise ValueError(
                    f" {self.n_features} features specified, but {self.features} has length {len(self.features)}"
                )

        # Mention source.
        if self.source is not None:
            self.source.mention()

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

    @property
    def balance(self) -> float:
        """
        Indicate how balanced the dataset is.
        """
        n_samples_per_class = [
            np.count_nonzero(self.labels == label) for label in range(self.n_classes)
        ]
        return min(n_samples_per_class) / max(n_samples_per_class)

    @property
    def feature_type(self) -> FeatureType:
        """
        The type of the features.
        """
        if self.features is None:
            return FeatureType.UNKNOWN
        else:
            feature_types = {feature.type for feature in self.features}
            if len(feature_types) == 1:
                return feature_types.pop()
            elif feature_types == {FeatureType.BINARY, FeatureType.CATEGORICAL}:
                return FeatureType.CATEGORICAL
            elif feature_types <= {
                FeatureType.BINARY,
                FeatureType.CATEGORICAL,
                FeatureType.NUMERICAL,
            }:
                return FeatureType.MIXED
            else:
                return FeatureType.UNKNOWN

    @abstractmethod
    def load(self):
        """
        Expected to fill self._points and self._targets,
        fit self.label_encoder, and set self.loaded.
        """

    def feature_types_match(self, raise_exception: bool) -> bool:
        if self.features is None:
            return True
        matches = [
            feature.type.matches(self.points[:, i_feature])
            for i_feature, feature in enumerate(self.features)
        ]
        if not all(matches) and raise_exception:
            raise FeatureTypeMismatch(
                f"Data does not match feature type for {self.features[matches.index(False)]}"
            )
        return all(matches)

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


class PublicDataset(Dataset, ABC):
    """
    Abstract base class for a published dataset that is available online.
    """

    # pylint: disable=useless-super-delegation
    # Annotations differ. Fixed in a future release of pylint.

    def __init__(
        self,
        *,
        safe_name: str,
        short_name: str,
        long_name: str,
        features: Optional[Sequence[Feature]] = None,
        n_samples: int,
        n_features: int,
        n_classes: int,
        source: Source,
    ):
        super().__init__(
            safe_name=safe_name,
            short_name=short_name,
            long_name=long_name,
            features=features,
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            source=source,
        )


class GenericDataset(Dataset):
    """
    Convenience class to load comma-separated values.
    """

    def __init__(
        self,
        path: str,
        *,
        safe_name: str,
        short_name: str,
        long_name: Optional[str] = None,
        features: Optional[Sequence[Feature]] = None,
        n_samples: Optional[int] = None,
        n_features: Optional[int] = None,
        n_classes: Optional[int] = None,
        source: Optional[Source] = None,
    ):
        if long_name is None:
            long_name = short_name
        super().__init__(
            safe_name=safe_name,
            short_name=short_name,
            long_name=long_name,
            features=features,
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            source=source,
        )
        self.path = path

    def load(self):
        # Load data.
        data = np.genfromtxt(self.path)
        points = data[:, :-1]
        labels = data[:, -1].astype(int)

        # Extract numbers.
        n_points, n_features = points.shape
        n_classes = len(np.unique(labels))

        # Set/Check numbers.
        if self.n_samples is None:
            self.n_samples = n_points
        else:
            assert self.n_samples == n_points
        if self.n_features is None:
            self.n_features = n_features
        else:
            assert self.n_features == n_features
        if self.n_classes is None:
            self.n_classes = n_classes
        else:
            assert self.n_classes == n_classes

        # Set data.
        self._points = points
        self._targets = self.label_encoder.fit_transform(labels)

        self.loaded = True
