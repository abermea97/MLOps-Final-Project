# Databricks notebook source
import mlflow
import mlflow.pyfunc

# COMMAND ----------

pip install --upgrade mlflow

# COMMAND ----------

# Set MLflow to use Unity Catalog
mlflow.set_registry_uri("databricks")

# COMMAND ----------

# Define model path (in DBFS) and model name
model_path = "/dbfs/databricks/mlflow-tracking/1303991687879297/969793f27af7446489a004f6163b7cca/artifacts/model"
model_name = "flops.default.CDC_Diabetes_Model"

# COMMAND ----------

# Register the model
result = mlflow.register_model(model_uri=model_path, name=model_name)

# Print the details of the registered model
print(f"Model registered with name: {result.name} and version: {result.version}")

# COMMAND ----------


