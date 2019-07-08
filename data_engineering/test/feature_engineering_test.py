import unittest
from data_engineering.components.feature_engineering import FeatureEngineering


class FeatureEngineeringTest(unittest.TestCase):
    def test1(self):
        self.assertRaises(Exception, FeatureEngineering().
                          process('raw_data.parquet', 'processing_data.parquet'))
