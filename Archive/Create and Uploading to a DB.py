# Databricks notebook source
# MAGIC %md
# MAGIC # Uploading the file into the dbfs (Databricks File System)
# MAGIC
# MAGIC This has to be done to use the automl features

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create a Database to store these tables

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE DATABASE IF NOT EXISTS diabetes_test;

# COMMAND ----------

df = pd.read_csv('diabetes_r.csv', encoding="utf-8", encoding_errors="replace")

print("Shape: ", df.shape)

# COMMAND ----------

import pandas as pd

# read csv
df = pd.read_csv('diabetes_r.csv', encoding="utf-8", encoding_errors="replace")
df = df.drop('Unnamed: 0', axis=1)
print("Shape: ", df.shape)

# convert to spark df and upload into the database
spark_df = spark.createDataFrame(df)
spark_df.write.saveAsTable("diabetes_r", mode="overwrite")

df.head(4)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Confirm the Table is in the DB

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM diabetes_r;

# COMMAND ----------

file_url = 'diabetes_r.csv'
dbfs_path = 'dbfs:/FileStore/tables/diabetes_r.csv'

dbutils.fs.cp(file_url, dbfs_path)

# COMMAND ----------

dbutils.fs.ls("dbfs:/FileStore/tables/")

# COMMAND ----------


