import unittest
from data_engineering.components.building_model import BuildingModel


class BuildingModelTest(unittest.TestCase):
    def test1(self):
        self.assertRaises(Exception, BuildingModel().
                          process('processing_data.parquet', 'prod_model'))

