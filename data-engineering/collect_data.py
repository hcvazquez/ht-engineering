# Databricks notebook source
#create sparksession object
from pyspark.sql import SparkSession
spark=SparkSession.builder.appName('handling_time').getOrCreate()

# COMMAND ----------

#import Linear Regression from spark's MLlib
from pyspark.ml.regression import LinearRegression

# COMMAND ----------

#Load the dataset
df=spark.read.csv('/FileStore/tables/data_handling.csv',inferSchema=True,header=True)

