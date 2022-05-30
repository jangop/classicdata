"""
USPS
"""
import tarfile

import numpy as np
import scipy.io

from ..dataset import Feature, FeatureType, PublicDataset, Source
from ..files import provide_file
from ..settings import base_directory


class USPS(PublicDataset):
    """
    Loader for USPS Handwritten Digits
    """

    def __init__(self):
        features = [
            Feature(
                short_name=f"({i}, {j})",
                long_name=f"Pixel ({i}, {j})",
                type=FeatureType.NUMERICAL,
                description=f"Pixel intensity at position ({i}, {j})",
            )
            for i in range(16)
            for j in range(16)
        ]
        super().__init__(
            safe_name="usps",
            short_name="USPS",
            long_name="USPS Handwritten Digits",
            features=features,
            n_samples=4649 * 2,
            n_features=256,
            n_classes=10,
            source=Source(
                name="Gaussian Processes for Machine Learning: Data",
                url="http://www.gaussianprocess.org/gpml/data/",
                citation_url="http://www.gaussianprocess.org/gpml/",
            ),
        )

    def load(self):
        # Obtain data.
        file_path = provide_file(
            url="http://www.gaussianprocess.org/gpml/data/usps_resampled.tar.bz2",
            root_directory=base_directory,
            sub_directory=self.safe_name,
            expected_hash=(
                "f93d1079fbcb82e10938298bbff9382d5a58bd76dd17cdbfe4c687261c7bae54"
                "fc4232c2a9cbbef6b1194dd3615f3bc8d77b95278658b50217c0f3c3fa5260db"
            ),
        )

        # Decompress.
        with tarfile.open(file_path) as directory:
            decompressed = directory.extractfile("usps_resampled/usps_resampled.mat")
            # Parse matlab format.
            mat = scipy.io.loadmat(decompressed)

            # Extract points and labels
            training_points = mat["train_patterns"].T
            training_labels = mat["train_labels"].T
            test_points = mat["test_patterns"].T
            test_labels = mat["test_labels"].T

            # Undo one-hot encoding.
            training_labels = np.nonzero(training_labels == 1)[1]
            test_labels = np.nonzero(test_labels == 1)[1]

            # Combine training and test data.
            self._points = np.vstack((training_points, test_points))
            labels = np.hstack((training_labels, test_labels))

            # Encode labels.
            self._targets = self.label_encoder.fit_transform(labels)

            self.loaded = True
