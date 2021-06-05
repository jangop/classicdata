"""
Test basic properties of implemented datasets.
"""
import unittest

import numpy as np

from classicdata import Ionosphere
from classicdata.dataset import Dataset


class TestLoading(unittest.TestCase):
    def test_loading(self):
        for DatasetImplementation in Dataset.__subclasses__():
            with self.subTest(Dataset=DatasetImplementation):
                dataset_instance = DatasetImplementation()

                # `.loaded` should not be set.
                self.assertFalse(dataset_instance.loaded)

                # Number of points and number of features should be correctly defined.
                self.assertEqual(
                    dataset_instance.points.shape,
                    (dataset_instance.n_samples, dataset_instance.n_features),
                )

                # Number of labels must be defined correctly.
                self.assertEqual(
                    dataset_instance.labels.shape, (dataset_instance.n_samples,)
                )

                # Convert labels “there and back”.
                recoded_labels = dataset_instance.label_encoder.transform(
                    dataset_instance.decode_labels(dataset_instance.labels)
                )
                self.assertTrue(np.all(recoded_labels == dataset_instance.labels))

                # `.loaded` should be set.
                self.assertTrue(dataset_instance.loaded)

                # Count random split.
                (
                    train_points,
                    train_labels,
                    test_points,
                    test_labels,
                ) = dataset_instance.split_for_training()
                self.assertEqual(
                    train_points.shape[0] + test_points.shape[0],
                    dataset_instance.n_samples,
                )
                self.assertEqual(
                    train_labels.shape[0] + test_labels.shape[0],
                    dataset_instance.n_samples,
                )

    def test_zero_test_split(self):
        dataset = Ionosphere()  # Arbitrarily chosen.
        dataset.load()
        (
            train_points,
            train_labels,
            test_points,
            test_labels,
        ) = dataset.split_for_training(test_size=0)
        self.assertEqual(train_points.shape[0], dataset.n_samples)
        self.assertEqual(train_labels.shape[0], dataset.n_samples)
        self.assertEqual(test_points.shape[0], 0)
        self.assertEqual(test_labels.shape[0], 0)


if __name__ == "__main__":
    unittest.main()
