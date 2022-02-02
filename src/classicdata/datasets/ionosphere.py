"""
Ionosphere
"""
import numpy as np

from ..dataset import Dataset
from ..files import provide_file
from ..settings import base_directory
from ..utility import translate, uci_ml_repo


class Ionosphere(Dataset):
    """
    Loader for the Ionosphere Dataset.
    """

    def __init__(self):
        super().__init__(
            safe_name="ionosphere",
            short_name="Ionosphere",
            long_name="Ionosphere",
            n_samples=351,
            n_features=34,
            n_classes=2,
            source=uci_ml_repo,
        )

    def load(self):
        """
        Load data.
        """
        data_path = provide_file(
            url=(
                "https://archive.ics.uci.edu/ml/machine-learning-databases/"
                "ionosphere/ionosphere.data"
            ),
            root_directory=base_directory,
            sub_directory=self.safe_name,
            expected_hash=(
                "dd4f8d72bdbe61314c9e9b1916c19c8ff32dae9c264ce01e2b97b06ccdb78058"
                "1cf8fa8a8058d78feafe62ea9797b21e94606591c18a020fb0920ac8284a8910"
            ),
        )

        # Load _points.
        self._points = np.genfromtxt(
            data_path, dtype=float, usecols=range(self.n_features), delimiter=","
        )

        # Load _targets
        labels = np.genfromtxt(
            data_path, dtype=str, usecols=(self.n_features,), delimiter=","
        )

        # Prettify _targets.
        labels = translate(labels, {"g": "good", "b": "bad"})

        # Encode _targets.
        self._targets = self.label_encoder.fit_transform(labels)

        self.loaded = True
