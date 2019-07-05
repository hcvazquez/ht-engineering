from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegressionModel

spark = SparkSession.builder.appName('handling_time_inference').getOrCreate()

#Load Linear Regression model 
lr_model = LinearRegressionModel.load('prod_model')

new_data = spark.read.parquet('new_data.parquet')

new_predictions = lr_model.evaluate(new_data)

#Save result as parquet
new_predictions.write.format("parquet").mode('overwrite').option("header", "true").save("new_data.parquet")
