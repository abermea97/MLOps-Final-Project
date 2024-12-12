# Databricks notebook source
# MAGIC %md
# MAGIC # Use AutoML Cluster 3 for this to run

# COMMAND ----------

# MAGIC %md # AutoML regression example
# MAGIC
# MAGIC ## Requirements
# MAGIC Databricks Runtime for Machine Learning 8.3 or above.

# COMMAND ----------

import pandas as pd
from databricks import automl
import json
import os

# COMMAND ----------

def train_model(df, target_feature):
  # Convert target column to IntegerType
  df[target_feature] = df[target_feature].astype(int)
  
  # Training model
  summary = automl.classify(df, target_col=target_feature, timeout_minutes=30)
  print(f"Summary Object: {summary}")

  # Save the best model
  best_model_path = summary.best_trial.model_path

  return summary, best_model_path

# COMMAND ----------

if __name__ == "__main__":
    # Set up widgets for parameter passing
    dbutils.widgets.text("processed_df_path", "")
    dbutils.widgets.text("target_feature", "")
    
    # Retrieve parameters
    processed_df_path = dbutils.widgets.get("processed_df_path")
    target_feature = dbutils.widgets.get("target_feature")

    # Load processed data
    df = pd.read_csv(processed_df_path)

    # Train model
    summary, best_model_path = train_model(df, target_feature)

    # Exit with results
    result = {
        "summary": str(summary),
        "best_model_path": best_model_path
    }
    dbutils.notebook.exit(json.dumps(result))

# COMMAND ----------


