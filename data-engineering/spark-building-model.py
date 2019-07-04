# Databricks notebook source
#create sparksession object
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('handling_time_building_model').getOrCreate()

# COMMAND ----------

#create data containing input features and output column
train_df = spark.read.parquet('processing_data.parquet').select('scaledFeatures','HT')

# COMMAND ----------

#import Linear Regression from spark's MLlib
from pyspark.ml.regression import LinearRegression

#Build Linear Regression model 
lin_Reg = LinearRegression(featuresCol="scaledFeatures", labelCol='HT')

# COMMAND ----------

#fit the linear regression model on training data set 
lr_model = lin_Reg.fit(train_df)

# COMMAND ----------

lr_model.save('prod_model')

# COMMAND ----------


