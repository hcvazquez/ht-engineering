from pyspark.sql import SparkSession


class CollectData:

    def __init__(self):
        """
        Class Initializer
        """
        self.spark = SparkSession.builder.appName('handling_time_collect_data').getOrCreate()

    def process(self, data_input, data_output):
        """
        An spark process to collect data
        :param data_input: data input filename
        :param data_output: data output filename
        """

        # Load the dataset
        df = self.spark.read.csv(data_input, inferSchema=True, header=True)

        # Save dataset as parquet
        df.write.format("parquet").mode('overwrite').save(data_output)


if __name__ == '__main__':
    CollectData().process('/FileStore/tables/data_handling.csv', 'raw_data.parquet')

