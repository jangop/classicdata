"""
Pen-Based Recognition of Handwritten Digits
"""
import numpy as np

from ..dataset import Dataset
from ..files import provide_file
from ..settings import base_directory
from ..utility import uci_ml_repo


class PenDigits(Dataset):
    """
    Loader for Pen-Based Recognition of Handwritten Digits.
    """

    def __init__(self):
        super().__init__(
            safe_name="pendigits",
            short_name="Pen Digits",
            long_name="Pen-Based Recognition of Handwritten Digits",
            n_samples=10992,
            n_features=16,
            n_classes=10,
            source=uci_ml_repo,
        )

    def load(self):
        """
        Load data.
        """

        def load_points_and_targets(file_path):
            data = np.genfromtxt(file_path, dtype=float, delimiter=",")
            points = data[:, :-1].astype(int)
            targets = data[:, -1].astype(int)

            return points, targets

        # Get the data, provided as separate files for training and test.
        training_path = provide_file(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/"
            "pendigits/pendigits.tra",
            root_directory=base_directory,
            sub_directory=self.safe_name,
            expected_hash=(
                "711e62b874e7ead9ac470e676ff47c0b51f5871f70ea9c2d014de632c39e68c3"
                "d71b6ce0c7e973133228f451fdfe6d81dc3201518dc39464a7e36d3145f8a928"
            ),
        )
        trainig_points, training_labels = load_points_and_targets(training_path)

        test_path = provide_file(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/"
            "pendigits/pendigits.tes",
            root_directory=base_directory,
            sub_directory=self.safe_name,
            expected_hash=(
                "bc7400e7dc2a01411db47f2c1dbb94d9a78c593ac49abb934c4c67933232285f"
                "04a99ea41894d76e4f944dcb6bf62d2abe0046866e7984e04990d5371290fc2d"
            ),
        )
        test_points, test_labels = load_points_and_targets(test_path)

        # Combine training and test data.
        self._points = np.vstack((trainig_points, test_points))
        labels = np.hstack((training_labels, test_labels))

        # Encode labels.
        self._targets = self.label_encoder.fit_transform(labels)

        self.loaded = True
