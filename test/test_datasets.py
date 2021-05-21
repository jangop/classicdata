"""
Test basic properties of implemented datasets.
"""
import unittest

from classicdata.dataset import Dataset


class TestLoading(unittest.TestCase):
    def test_loading(self):
        for DatasetImplementation in Dataset.__subclasses__():
            with self.subTest(Dataset=DatasetImplementation):
                dataset_instance = DatasetImplementation()
                dataset_instance.load()
                self.assertTrue(dataset_instance.loaded)
                self.assertEqual(
                    dataset_instance.points.shape,
                    (dataset_instance.n_samples, dataset_instance.n_features),
                )
                self.assertEqual(
                    dataset_instance.labels.shape, (dataset_instance.n_samples,)
                )


if __name__ == "__main__":
    unittest.main()
