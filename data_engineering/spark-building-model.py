from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression

spark = SparkSession.builder.appName('handling_time_building_model').getOrCreate()

# create data containing input features and output column
train_df = spark.read.parquet('processing_data.parquet').select('scaledFeatures', 'HT')

# Build Linear Regression model
lin_Reg = LinearRegression(featuresCol="scaledFeatures", labelCol='HT')

# fit the linear regression model on training data set
lr_model = lin_Reg.fit(train_df)

lr_model.save('prod_model')


