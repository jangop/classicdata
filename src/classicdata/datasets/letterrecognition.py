"""
Letter Recognition
"""
import numpy as np

from ..dataset import Dataset
from ..files import provide_file
from ..settings import base_directory
from ..utility import uci_ml_repo


class LetterRecognition(Dataset):
    """
    Loader for Letter Recognition.
    """

    def __init__(self):
        super().__init__(
            safe_name="letterrecognition",
            short_name="Letter Recognition",
            long_name="Letter Recognition",
            n_samples=20000,
            n_features=16,
            n_classes=10,
            source=uci_ml_repo,
        )

    def load(self):
        """
        Load data.
        """
        data_path = provide_file(
            url="https://archive.ics.uci.edu/ml/machine-learning-databases/"
            "letter-recognition/letter-recognition.data",
            root_directory=base_directory,
            sub_directory=self.safe_name,
            expected_hash=(
                "a1c18afc749d345a441cd056cd8a9cb0ccda350b1192db1a01a242c55514ce86"
                "f1c998aa6241f94232cdb8b3f34ee4721f832e79a273bf0059f3ee4673cee0a2"
            ),
        )

        self._points = np.genfromtxt(
            data_path,
            dtype=int,
            usecols=range(1, self.n_features + 1),
            delimiter=",",
        )
        labels = np.genfromtxt(data_path, dtype=str, usecols=(0,), delimiter=",")

        # Encode _targets.
        self._targets = self.label_encoder.fit_transform(labels)

        self.loaded = True
