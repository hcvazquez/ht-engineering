from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression


class BuildingModel:

    def __init__(self):
        """
        Class Initializer
        """
        self.spark = SparkSession.builder.appName('handling_time_building_model').getOrCreate()

    def process(self, data_input, data_output):
        """
        An spark process to build a model
        :param data_input: data input filename
        :param data_output: data output filename
        """

        # create data containing input features and output column
        train_df = self.spark.read.parquet(data_input).select('scaledFeatures', 'HT')

        # Build Linear Regression model
        lin_Reg = LinearRegression(featuresCol="scaledFeatures", labelCol='HT')

        # fit the linear regression model on training data set
        lr_model = lin_Reg.fit(train_df)

        lr_model.save(data_output)


if __name__ == '__main__':
    BuildingModel().process('processing_data.parquet', 'prod_model')
