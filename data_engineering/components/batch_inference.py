from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegressionModel


class BatchInference:

    def __init__(self):
        """
        Class Initializer
        """
        self.spark = SparkSession.builder.appName('handling_time_inference').getOrCreate()

    def process(self, data_input, data_output, model):
        """
        An spark process to do inference
        :param data_input: data input filename
        :param data_output: data output filename
        """

        # Load Linear Regression model
        lr_model = LinearRegressionModel.load(model)

        new_data = self.spark.read.parquet(data_input)

        new_predictions = lr_model.evaluate(new_data)

        # Save result as parquet
        new_predictions.write.format("parquet").mode('overwrite').option("header", "true").save(data_output)


if __name__ == '__main__':
    BatchInference().process('new_data.parquet', 'new_prediction.parquet', 'prod_model')
