"""
Wall-Following Robot Navigation
"""
import numpy as np

from ..dataset import Dataset
from ..files import provide_file
from ..settings import base_directory
from ..utility import uci_ml_repo


class RobotNavigation(Dataset):
    """
    Loader for Wall-Following Robot Navigation Data dataset.
    """

    def __init__(self):
        super().__init__(
            safe_name="robnav",
            short_name="Robot Navigation",
            long_name="Wall-Following Robot Navigation",
            n_samples=5456,
            n_features=24,
            n_classes=4,
            source=uci_ml_repo,
        )

    def load(self):
        """
        Load data.
        """
        data_path = provide_file(
            url="https://archive.ics.uci.edu/ml/machine-learning-databases/"
            "00194/sensor_readings_24.data",
            root_directory=base_directory,
            sub_directory=self.safe_name,
            expected_hash=(
                "02cc6c2d383bff1f701205cf06de1c115c4639ffae172ec6bc1ae118d1baf2d4"
                "96f1967e0de8ecd19bf90d05649143b6c3362fd990d4e1b9d74ae24860abafe4"
            ),
        )

        self._points = np.genfromtxt(
            data_path, dtype=np.float, usecols=range(self.n_features), delimiter=","
        )
        labels = np.genfromtxt(
            data_path, dtype=str, usecols=(self.n_features,), delimiter=","
        )

        # Encode _targets.
        self._targets = self.label_encoder.fit_transform(labels)

        self.loaded = True
