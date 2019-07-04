# Databricks notebook source
#create sparksession object
from pyspark.sql import SparkSession
spark=SparkSession.builder.appName('handling_time_collect_data').getOrCreate()

# COMMAND ----------

#Load the dataset
df=spark.read.csv('/FileStore/tables/data_handling.csv',inferSchema=True,header=True)

# COMMAND ----------

#Save dataset as parquet
df.write.format("parquet").mode('overwrite').save("raw_data.parquet")
