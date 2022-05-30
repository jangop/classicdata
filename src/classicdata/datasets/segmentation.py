"""
Image Segmentation
"""
import numpy as np

from ..dataset import Feature, FeatureType, PublicDataset
from ..files import provide_file
from ..settings import base_directory
from ..utility import uci_ml_repo


class ImageSegmentation(PublicDataset):
    """
    Loader for the Image Segmentation dataset.
    """

    def __init__(self):
        features = [
            Feature(
                short_name="centroid-col",
                long_name="Centroid Column",
                type=FeatureType.NUMERICAL,
                description="The column of the centroid.",
            ),
            Feature(
                short_name="centroid-row",
                long_name="Centroid Row",
                type=FeatureType.NUMERICAL,
                description="The row of the centroid.",
            ),
            Feature(
                short_name="vertical-edge-mean",
                long_name="Vertical Edge Mean",
                type=FeatureType.NUMERICAL,
                description="The mean of the contrast of horizontally adjacent pixels.",
            ),
            Feature(
                short_name="vertical-edge-std",
                long_name="Vertical Edge Standard Deviation",
                type=FeatureType.NUMERICAL,
                description="The standard deviation of the contrast of horizontally adjacent pixels.",
            ),
            Feature(
                short_name="horizontal-edge-mean",
                long_name="Horizontal Edge Mean",
                type=FeatureType.NUMERICAL,
                description="The mean of the contrast of vertically adjacent pixels.",
            ),
            Feature(
                short_name="horizontal-edge-std",
                long_name="Horizontal Edge Standard Deviation",
                type=FeatureType.NUMERICAL,
                description="The standard deviation of the contrast of vertically adjacent pixels.",
            ),
            Feature(
                short_name="intensity-mean",
                long_name="Intensity Mean",
                type=FeatureType.NUMERICAL,
                description="The mean of the intensity of the pixels.",
            ),
            Feature(
                short_name="red-mean",
                long_name="Red Mean",
                type=FeatureType.NUMERICAL,
                description="The mean of the red channel of the pixels.",
            ),
            Feature(
                short_name="blue-mean",
                long_name="Blue Mean",
                type=FeatureType.NUMERICAL,
                description="The mean of the blue channel of the pixels.",
            ),
            Feature(
                short_name="green-mean",
                long_name="Green Mean",
                type=FeatureType.NUMERICAL,
                description="The mean of the green channel of the pixels.",
            ),
            Feature(
                short_name="excess-red-mean",
                long_name="Excess Red Mean",
                type=FeatureType.NUMERICAL,
                description="The mean of the red channel of the pixels minus the mean of the green and blue channel of the pixels.",
            ),
            Feature(
                short_name="excess-blue-mean",
                long_name="Excess Blue Mean",
                type=FeatureType.NUMERICAL,
                description="The mean of the blue channel of the pixels minus the mean of the red and green channel of the pixels.",
            ),
            Feature(
                short_name="excess-green-mean",
                long_name="Excess Green Mean",
                type=FeatureType.NUMERICAL,
                description="The mean of the green channel of the pixels minus the mean of the red and blue channel of the pixels.",
            ),
            Feature(
                short_name="value-mean",
                long_name="Value Mean",
                type=FeatureType.NUMERICAL,
            ),
            Feature(
                short_name="saturation-mean",
                long_name="Saturation Mean",
                type=FeatureType.NUMERICAL,
            ),
            Feature(
                short_name="hue-mean", long_name="Hue Mean", type=FeatureType.NUMERICAL
            ),
        ]
        super().__init__(
            safe_name="segmentation",
            short_name="Segmentation",
            long_name="Image Segmentation",
            features=features,
            n_samples=2310,
            n_features=16,
            n_classes=7,
            source=uci_ml_repo,
        )

    def load(self):
        """
        Load data.
        """

        # Leave out three features as suggested in the literature,
        # and skip the first 5 lines, because they contain headers.
        # The first column holds the labels as uppercase strings.
        def load_points_and_targets(file_path):
            # Load points.
            points = np.genfromtxt(
                file_path,
                dtype=float,
                delimiter=",",
                skip_header=5,
                usecols=(1, 2) + tuple(range(6, 20)),
            )

            # Load labels.
            targets = np.genfromtxt(
                file_path,
                dtype=str,
                delimiter=",",
                skip_header=5,
                usecols=(0,),
                encoding=None,
            )

            return points, targets

        # Get the data, provided as separate files for training and test.
        training_path = provide_file(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/"
            "image/segmentation.data",
            root_directory=base_directory,
            sub_directory=self.safe_name,
            expected_hash=(
                "3e0abef520308ec0204e1c097e4402487f69ebe6e576b2e9caae514be042902e"
                "efe3c2dc127a584753c4084e93a2afd2ab830aa97703eef4903cc4f3ef63da7b"
            ),
        )
        trainig_points, training_labels = load_points_and_targets(training_path)

        test_path = provide_file(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/"
            "image/segmentation.test",
            root_directory=base_directory,
            sub_directory=self.safe_name,
            expected_hash=(
                "30c90676f4e31e9433ae4247ec2b554c7139b3265e92a3710ba04f7e8e70aa18"
                "89821f45fffb89ac058f5d3091fe48ced9ae92e8c110d3e55b18215351a6285c"
            ),
        )
        test_points, test_labels = load_points_and_targets(test_path)

        # Combine training and test data.
        self._points = np.vstack((trainig_points, test_points))
        labels = np.hstack((training_labels, test_labels))

        # Prettify labels.
        labels = np.ascontiguousarray([label.lower() for label in labels])

        # Encode labels.
        self._targets = self.label_encoder.fit_transform(labels)

        self.loaded = True
