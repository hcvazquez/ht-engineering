# Databricks notebook source
#create sparksession object
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('handling_time_inference').getOrCreate()

# COMMAND ----------

#import Linear Regression Model from spark's MLlib
from pyspark.ml.regression import LinearRegressionModel

#Load Linear Regression model 
lr_model = LinearRegressionModel.load('prod_model')

# COMMAND ----------

new_predictions = lr_model.evaluate(new_data)
