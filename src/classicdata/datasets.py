"""
Actual classic datasets.
"""

from typing import Optional

import numpy as np
from sklearn.preprocessing import LabelEncoder

from .dataset import Dataset
from .files import provide_file
from .settings import base_directory


class Ionosphere(Dataset):
    """
    Loader for the Ionosphere Dataset.
    """

    def __init__(self):
        super().__init__("ionosphere", "Ionosphere", "Ionosphere")

        self.n_samples = 351
        self.n_features = 34
        self.n_classes = 2

    def load(self):
        """
        Load data.
        """
        data_path = provide_file(
            url="https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data",
            root_directory=base_directory,
            sub_directory=self.safe_name,
            expected_hash="dd4f8d72bdbe61314c9e9b1916c19c8ff32dae9c264ce01e2b97b06ccdb780581cf8fa8a8058d78feafe62ea9797b21e94606591c18a020fb0920ac8284a8910",
        )

        # Load points.
        self.points = np.genfromtxt(
            data_path, dtype=float, usecols=range(self.n_features), delimiter=","
        )

        # Load labels
        labels = np.genfromtxt(
            data_path, dtype=str, usecols=(self.n_features,), delimiter=","
        )

        # Encode labels.
        self.labels = self.label_encoder.fit_transform(labels)

        self.loaded = True


class MagicGammaTelescope(Dataset):
    """
    Loader for the MAGIC Gamma Telescope Dataset.
    """

    def __init__(self):
        super().__init__("magic", "Telescope", "MAGIC Gamma Telescope")

        self.n_samples = 19020
        self.n_features = 10
        self.n_classes = 2

    def load(self):
        """
        Load data.
        """
        data_path = provide_file(
            url="https://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data",
            root_directory=base_directory,
            sub_directory=self.safe_name,
            expected_hash="1179fbcd4e71814aa3131bbb314c2fb702e5d475e7874346403c9c522f293798ae3717fbc689a0d64fc11e8b3e797cb8acb25f601ca12107d5a664cbbc506b1b",
        )

        # Load points.
        self.points = np.genfromtxt(
            data_path, dtype=float, usecols=range(self.n_features), delimiter=","
        )

        # Load labels
        labels = np.genfromtxt(
            data_path, dtype=str, usecols=(self.n_features,), delimiter=","
        )

        # Encode labels.
        self.labels = self.label_encoder.fit_transform(labels)

        self.loaded = True
