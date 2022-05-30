import unittest

import numpy as np

from classicdata.dataset import (
    Feature,
    FeatureType,
    FeatureTypeMismatch,
    FeatureTypeWarning,
)


class TestFeatures(unittest.TestCase):
    def test_features(self):
        binary_feature = Feature(
            short_name="Binary", long_name="Binary Feature", type=FeatureType.BINARY
        )
        categorical_feature = Feature(
            short_name="Categorical",
            long_name="Categorical Feature",
            type=FeatureType.CATEGORICAL,
        )
        ordinal_feature = Feature(
            short_name="Ordinal", long_name="Ordinal Feature", type=FeatureType.ORDINAL
        )
        numerical_feature = Feature(
            short_name="Numerical",
            long_name="Numerical Feature",
            type=FeatureType.NUMERICAL,
        )

        binary_data = np.array([0] * 50 + [1] * 50)
        categorical_data = np.array([0] * 50 + [1] * 50 + [2] * 50)
        numerical_data = np.linspace(-10, 10, 10000)

        self.assertTrue(binary_feature.type.matches(binary_data))
        self.assertFalse(binary_feature.type.matches(categorical_data))
        self.assertFalse(binary_feature.type.matches(numerical_data))

        self.assertTrue(categorical_feature.type.matches(categorical_data))
        with self.assertWarns(FeatureTypeWarning):
            categorical_feature.type.matches(numerical_data)

        self.assertTrue(ordinal_feature.type.matches(categorical_data))
        with self.assertWarns(FeatureTypeWarning):
            ordinal_feature.type.matches(numerical_data)

        self.assertTrue(numerical_feature.type.matches(numerical_data))
        with self.assertWarns(FeatureTypeWarning):
            numerical_feature.type.matches(binary_data)
            numerical_feature.type.matches(categorical_data)


if __name__ == "__main__":
    unittest.main()
