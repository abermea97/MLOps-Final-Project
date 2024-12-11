# Databricks notebook source
# MAGIC %md
# MAGIC ## Requirements
# MAGIC Databricks Runtime for Machine Learning 8.3 or above.

# COMMAND ----------

import pandas as pd
from databricks import automl
import json
import os
import mlflow

# COMMAND ----------

# Defining the function to train a model and log outputs to MLflow
def train_model(df, target_feature):
    # Converting the target column to IntegerType
    df[target_feature] = df[target_feature].astype(int)
    
    # Running the model training using Databricks AutoML
    summary = automl.classify(
        dataset=df, 
        target_col=target_feature, 
        timeout_minutes=40
    )

    # Retrieving the best model path and ID from AutoML results
    best_model_path = summary.best_trial.model_path
    best_run_id = summary.best_trial.mlflow_run_id

    return summary, best_model_path, best_run_id

# COMMAND ----------

# Main execution
if __name__ == "__main__":
    # Setting up widgets for parameter passing
    dbutils.widgets.text("processed_df_path", "")
    dbutils.widgets.text("target_feature", "")
    
    # Retrieving parameters
    processed_df_path = dbutils.widgets.get("processed_df_path")
    target_feature = dbutils.widgets.get("target_feature")

    # Loading processed data
    df = pd.read_csv(processed_df_path)

    # Training the model and logging results
    summary, best_model_path, best_run_id = train_model(df, target_feature)

    # Exiting with results
    result = {
        "summary": str(summary),
        "best_model_path": best_model_path,
        "best_run_id": best_run_id
    }

    # Ending the nested run automatically when the block exits
    dbutils.notebook.exit(json.dumps(result))
