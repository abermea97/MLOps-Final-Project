# Databricks notebook source
# MAGIC %md 
# MAGIC # Pipeline
# MAGIC Note: Use AutoML Cluster 3

# COMMAND ----------

# MAGIC %md
# MAGIC ## Defining pipeline function

# COMMAND ----------

import json
import mlflow
from mlflow.tracking import MlflowClient
import os
import shutil
from io import StringIO

def pipeline(path, target_feature, experiment_name="Test", test_size=0.2, random_state=123):
    # Ending any potential previous runs
    mlflow.end_run()

    # Setting the MLflow experiment
    mlflow.set_experiment(experiment_name)

    # Starting the MLflow run
    with mlflow.start_run() as data_loading_run:
        data_loading_run_id = data_loading_run.info.run_id
        
        # Running the EDA Notebook
        loading_result = dbutils.notebook.run(
            "/Workspace/Users/abermea@uchicago.edu/Test/load_diabetes_data",
            timeout_seconds=600,
            arguments={
                "path": path,
                "target_feature": target_feature,
                "test_size": str(test_size),
                "random_state": str(random_state),
                "run_id": data_loading_run_id
            }
        )
        loading_result = json.loads(loading_result)  
        train_df_path = loading_result["train_df_path"]
        test_df_path = loading_result["test_df_path"]

        # ending data_loading_run
        mlflow.end_run()

    # Running the Modeling Notebook
    modeling_result = dbutils.notebook.run(
        "/Workspace/Users/abermea@uchicago.edu/Test/find_best_model",
        timeout_seconds=2400,
        arguments={
            "processed_df_path": train_df_path,
            "target_feature": target_feature
        }
    )
    modeling_result = json.loads(modeling_result)

    # Retrieving the modeling run ID
    modeling_run_id = modeling_result["best_run_id"]

    # Initializing the MLflow client
    client = MlflowClient()

    # Fetching the modeling run details
    modeling_run = client.get_run(modeling_run_id)

    # Logging contents from the modeling run into the data loading run
    with mlflow.start_run(run_id=data_loading_run_id):  # Reattaching to the data loading run
        # Copying metrics
        for key, value in modeling_run.data.metrics.items():
            mlflow.log_metric(key, value)

        # Copying parameters
        for key, value in modeling_run.data.params.items():
            mlflow.log_param(key, value)

        # Copying artifacts
        temp_artifact_path = f"/dbfs/tmp/artifacts_{modeling_run_id}"  # Temporary storage path

        # Ensuring the temporary directory exists
        if not os.path.exists(temp_artifact_path):
            os.makedirs(temp_artifact_path)

        client.download_artifacts(modeling_run_id, "", temp_artifact_path)

        for root, _, files in os.walk(temp_artifact_path):
            for file in files:
                full_file_path = os.path.join(root, file)
                artifact_relative_path = os.path.relpath(full_file_path, temp_artifact_path)
                mlflow.log_artifact(full_file_path, artifact_path=f"Modeling_Artifacts/{artifact_relative_path}")

        # Cleaning up temporary files
        shutil.rmtree(temp_artifact_path)

        # Log metrics from the source run to the target run
        for key, value in modeling_run.data.metrics.items():
            mlflow.log_metric(key, value)

        # Capturing Modeling Results as a text artifact
        text_output = StringIO()
        text_output.write("Modeling Results Summary:\n\n")
        text_output.write(modeling_result["summary"])

        text_output_path = "modeling_results_output.txt"
        with open(text_output_path, "w") as f:
            f.write(text_output.getvalue())
        text_output.close()

        mlflow.log_artifact(text_output_path, artifact_path="Modeling")
        os.remove(text_output_path)

    # Returning the final results
    return {
        "summary": modeling_result["summary"],
        "best_model_path": modeling_result["best_model_path"],
        "test_df_path": test_df_path
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## Running Pipeline

# COMMAND ----------

# Defining the parameters
path = "/dbfs/FileStore/tables/diabetes_raw.csv"
target_feature = "Diabetes_binary"
experiment_name = "/Users/jabautista@uchicago.edu/cdc_diabetes_model"

# Running the pipeline
results = pipeline(path, target_feature, experiment_name)

print("Model Training Summary:", results["summary"])
print("Best Model Saved At:", results["best_model_path"])
print("Test Data Saved At:", results["test_df_path"])
