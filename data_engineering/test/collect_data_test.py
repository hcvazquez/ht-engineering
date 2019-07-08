import unittest
from data_engineering.components.collect_data import CollectData


class CollectDataTest(unittest.TestCase):
    def test1(self):
        self.assertRaises(Exception, CollectData().
                          process('/FileStore/tables/data_handling.csv', 'raw_data.parquet'))
