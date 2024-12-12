# Databricks notebook source
# MAGIC %md 
# MAGIC # Pipeline
# MAGIC Note: Use AutoML Cluster 3

# COMMAND ----------

# MAGIC %md
# MAGIC ## Defining pipeline function

# COMMAND ----------

import json

def pipeline(path, target_feature, test_size=0.2, random_state=123):
    # Run EDA Notebook
    eda_result = dbutils.notebook.run(
        "/Workspace/Users/abermea@uchicago.edu/Test/eda",
        timeout_seconds=600,
        arguments={
            "path": path,
            "target_feature": target_feature,
            "test_size": str(test_size),  # Pass as string for widgets
            "random_state": str(random_state)
        }
    )
    eda_result = json.loads(eda_result)  # Parse the JSON result
    train_df_path = eda_result["train_df_path"]
    test_df_path = eda_result["test_df_path"]

    # Run Modeling Notebook
    modeling_result = dbutils.notebook.run(
        "/Workspace/Users/abermea@uchicago.edu/Test/modeling",
        timeout_seconds=2400,
        arguments={
            "processed_df_path": train_df_path,
            "target_feature": target_feature
        }
    )
    modeling_result = json.loads(modeling_result)  # Parse the JSON result

    # Return the final results, including the test data path for evaluation
    return {
        "summary": modeling_result["summary"],
        "best_model_path": modeling_result["best_model_path"],
        "test_df_path": test_df_path  # Include test_df_path for evaluation
    }


# COMMAND ----------

# Example usage
path = "/dbfs/FileStore/tables/diabetes_raw.csv"
target_feature = "Diabetes_binary"
results = pipeline(path, target_feature)

print("Model Training Summary:", results["summary"])
print("Best Model Saved At:", results["best_model_path"])
print("Test Data Saved At:", results["test_df_path"])

# COMMAND ----------


