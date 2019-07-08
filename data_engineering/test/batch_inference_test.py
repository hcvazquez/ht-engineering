import unittest
from data_engineering.components.batch_inference import BatchInference


class BatchInferenceTest(unittest.TestCase):
    def test1(self):
        self.assertRaises(Exception, BatchInference().
                          process('new_data.parquet', 'new_prediction.parquet', 'prod_model'))

