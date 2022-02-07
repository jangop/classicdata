"""
MAGIC Gamma Telescope
"""
import numpy as np

from ..dataset import Dataset
from ..files import provide_file
from ..settings import base_directory
from ..utility import translate, uci_ml_repo


class MagicGammaTelescope(Dataset):
    """
    Loader for the MAGIC Gamma Telescope Dataset.
    """

    def __init__(self):
        super().__init__(
            safe_name="magic",
            short_name="Telescope",
            long_name="MAGIC Gamma Telescope",
            n_samples=19020,
            n_features=10,
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
                "magic/magic04.data"
            ),
            root_directory=base_directory,
            sub_directory=self.safe_name,
            expected_hash=(
                "1179fbcd4e71814aa3131bbb314c2fb702e5d475e7874346403c9c522f293798"
                "ae3717fbc689a0d64fc11e8b3e797cb8acb25f601ca12107d5a664cbbc506b1b"
            ),
        )

        # Load points.
        self._points = np.genfromtxt(
            data_path, dtype=float, usecols=range(self.n_features), delimiter=","
        )

        # Load targets
        labels = np.genfromtxt(
            data_path, dtype=str, usecols=(self.n_features,), delimiter=","
        )

        # Prettify _targets.
        labels = translate(labels, {"g": "gamma", "h": "hadron"})

        # Encode _targets.
        self._targets = self.label_encoder.fit_transform(labels)

        self.loaded = True
