# Databricks notebook source
#pip install evidently==0.2.0

# COMMAND ----------

#%pip install mlflow

# COMMAND ----------

#%pip install numpy==1.23.5

# COMMAND ----------

#%pip install scikit-learn==0.24.1

# COMMAND ----------

import evidently
import pandas as pd
import numpy as np

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Model 

# COMMAND ----------

import mlflow.pyfunc

# Load the model
model_path = "dbfs:/databricks/mlflow-tracking/1303991687879297/969793f27af7446489a004f6163b7cca/artifacts/model"
model = mlflow.pyfunc.load_model(model_path)

# COMMAND ----------

dbutils.fs.ls("/FileStore/tables")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data

# COMMAND ----------

train_data = pd.read_csv("/dbfs/FileStore/tables/train_data.csv")

# COMMAND ----------

test_data = pd.read_csv("/dbfs/FileStore/tables/test_data.csv")
test_data.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validate Results

# COMMAND ----------

# Drop the target column (if it exists) to get only features
X_test = test_data.drop(columns=["Diabetes_binary"], errors="ignore")

# Predict using the loaded model
y_pred = model.predict(X_test)

# Add predictions to the test data for evaluation
test_data["Predictions"] = y_pred


# COMMAND ----------

from sklearn.metrics import f1_score, classification_report

# Assuming 'Diabetes_binary' is your true label column
y_true = test_data["Diabetes_binary"]
y_pred = test_data["Predictions"]


# classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Monitoring with Evidently AI 

# COMMAND ----------

from evidently.dashboard import Dashboard
from evidently.tabs import DataDriftTab, ClassificationPerformanceTab


# COMMAND ----------

train_data.head()

# COMMAND ----------

test_data.head()

# COMMAND ----------

# Ensure `train_data` has all the required features for prediction
X_train = train_data.drop(columns=["Diabetes_binary"], errors="ignore") 

# Generate predictions using the trained model
train_predictions = model.predict(X_train)

# Add predictions to the train_data
train_data["prediction"] = train_predictions


# COMMAND ----------

# Rename 'Diabetes_binary' to 'target' in train_data and test_data
train_data.rename(columns={"Diabetes_binary": "target"}, inplace=True)
test_data.rename(columns={"Diabetes_binary": "target"}, inplace=True)
test_data.rename(columns={"Predictions": "prediction"}, inplace=True)

# Verify the changes
print("Train Data Columns:", train_data.columns)
print("Test Data Columns:", test_data.columns)

# COMMAND ----------

from evidently.dashboard import Dashboard
from evidently.tabs import DataDriftTab, ClassificationPerformanceTab

# Create the Evidently dashboard
dashboard = Dashboard(tabs=[DataDriftTab(), ClassificationPerformanceTab()])

# Calculate metrics for monitoring
dashboard.calculate(train_data, test_data)

# Save the Evidently dashboard to an HTML file
dashboard.save("/dbfs/FileStore/evidently_dashboard.html")

# COMMAND ----------

# MAGIC %md
# MAGIC Download link: https://3715126558473529.9.gcp.databricks.com/files/evidently_dashboard.html

# COMMAND ----------

# MAGIC %md
# MAGIC ## Change 2 features: Swap Smoke and Stroke columns

# COMMAND ----------

print(test_data['Smoker'].value_counts())
print(test_data['Stroke'].value_counts())

# COMMAND ----------

test_data[['Smoker', 'Stroke']].hist(figsize=(10, 5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Swap Smoker and Stroke columns
# MAGIC By swapping the stroke and smoker columns, we will create a dataset with mostly smokers and even dsitribution of stroke cases

# COMMAND ----------

#copy test data 
test_data_swap = test_data.copy()

# COMMAND ----------

#drop prediction column
test_data_swap = test_data_swap.drop(columns=['prediction'])
test_data_swap.head()

# COMMAND ----------

test_data_swap[["Smoker", "Stroke"]] = test_data_swap[["Stroke", "Smoker"]]

# COMMAND ----------

print(test_data_swap['Smoker'].value_counts())
print(test_data_swap['Stroke'].value_counts())

# COMMAND ----------

# Drop the target column (if it exists) to get only features
X_test_swap = test_data_swap.drop(columns=["target"], errors="ignore")

# Predict using the loaded model
y_pred_swap = model.predict(X_test_swap)

# Add predictions to the test data for evaluation
test_data_swap["prediction"] = y_pred_swap


# COMMAND ----------

# Assuming 'Diabetes_binary' is your true label column
y_true_swap = test_data_swap["target"]
y_pred_swap = test_data_swap["prediction"]


# classification report
print("\nClassification Report:")
print(classification_report(y_true_swap, y_pred_swap))

# COMMAND ----------

from evidently.dashboard import Dashboard
from evidently.tabs import DataDriftTab, ClassificationPerformanceTab

# Create the Evidently dashboard
dashboard_2 = Dashboard(tabs=[DataDriftTab(), ClassificationPerformanceTab()])

# Calculate metrics for monitoring
dashboard_2.calculate(train_data, test_data_swap)

# Save the Evidently dashboard to an HTML file
dashboard_2.save("/dbfs/FileStore/evidently_dashboard_smoke_stroke_swap.html")

# COMMAND ----------

# MAGIC %md
# MAGIC Download link: https://3715126558473529.9.gcp.databricks.com/files/evidently_dashboard_smoke_stroke_swap.html
