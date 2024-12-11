# Databricks notebook source
!pip install imblearn

# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import mlflow
from imblearn.under_sampling import NearMiss
from sklearn.model_selection import train_test_split
from io import StringIO

# COMMAND ----------

# Defines function to load data and returns it in a pandas df
def load_data(path):
    df = pd.read_csv(path, encoding="utf-8")
    return df

# COMMAND ----------

def validate_data(df, target_feature):
    # Initializing a StringIO object to capture text outputs
    text_output = StringIO()

    # Logging general dataframe info
    text_output.write("Pre-processed Dataframe Validation:\n\n")
    text_output.write("\nDataframe info:\n")
    df.info(buf=text_output)

    # Logging nulls per column
    text_output.write("\n\nNulls per Column:\n")
    text_output.write(df.isna().sum().to_string())

    # Logging target feature distribution plot
    labels = df[target_feature].unique()
    labels = [str(label) for label in labels]
    df[target_feature].value_counts(1).plot(kind='barh', figsize=(10, 2)).spines[['top', 'right']].set_visible(False)
    plt.title(f'{target_feature} Distribution (%)', fontsize=18)
    plt.yticks(ticks=range(len(labels)), labels=labels)
    plt.tight_layout()
    target_dist_path = "target_distribution.png"
    plt.savefig(target_dist_path)
    plt.close()
    text_output.write(f"\n\nSaving target feature distribution plot.")

    # Logging correlation matrix plot
    corr_matrix = df.corr()
    plt.figure(figsize=(16, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    corr_matrix_path = "correlation_matrix.png"
    plt.savefig(corr_matrix_path)
    plt.close()
    text_output.write(f"\n\nSaving correlation matrix plot.")

    # Logging binary feature distributions
    text_output.write("\n\nBinary Feature Distribution:\n")
    bool_vars = (df.nunique()[df.nunique() == 2]
                  .index
                  .drop(labels=target_feature))
    for col in bool_vars:
        binary_dist_path = f"binary_feature_distribution_{col}.png"
        (df.groupby(target_feature)[col]
         .value_counts(1)
         .unstack()
         .iloc[:, ::-1]
         .plot(kind='barh', stacked=True, figsize=(10, 2), alpha=1)
         .spines[['top', 'right']].set_visible(False))
        plt.legend(['Yes', "No"], bbox_to_anchor=(1, 1, 0, 0), shadow=False, frameon=False)
        plt.yticks(ticks=[0, 1], labels=['Non-Diabetic', 'Diabetic'])
        plt.tight_layout()
        plt.title(col, fontsize=18)
        plt.savefig(binary_dist_path)
        plt.close()
        text_output.write(f"\nSaving binary feature distribution for {col}.\n")

    # Logging numeric feature distributions
    text_output.write("\n\nNumeric Feature Distribution:\n")
    num_vars = [var for var in df.columns if var not in bool_vars and var != target_feature]
    for var in num_vars:
        num_dist_path = f"numeric_feature_distribution_{var}.png"
        plt.figure()
        df[df[target_feature] == 0][var].hist(alpha=0.5, label='Diabetes=0', bins=30)
        df[df[target_feature] == 1][var].hist(alpha=0.5, label='Diabetes=1', bins=30)
        plt.title(var)
        plt.xlabel(var)
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        plt.savefig(num_dist_path)
        plt.close()
        text_output.write(f"\nSaving numeric feature distribution for {var}.\n")

    # Saving the text output to a file
    text_output_path = "eda_output.txt"
    with open(text_output_path, "w") as f:
        f.write(text_output.getvalue())
    text_output.close()

    # Logging all outputs to MLFlow
    mlflow.log_artifact(text_output_path, artifact_path="EDA")
    mlflow.log_artifact(target_dist_path, artifact_path="EDA/Plots")
    mlflow.log_artifact(corr_matrix_path, artifact_path="EDA/Plots")
    for col in bool_vars:
        mlflow.log_artifact(f"binary_feature_distribution_{col}.png", artifact_path="EDA/Plots")
    for var in num_vars:
        mlflow.log_artifact(f"numeric_feature_distribution_{var}.png", artifact_path="EDA/Plots")

    # Cleaning up local files
    os.remove(text_output_path)
    os.remove(target_dist_path)
    os.remove(corr_matrix_path)
    for col in bool_vars:
        os.remove(f"binary_feature_distribution_{col}.png")
    for var in num_vars:
        os.remove(f"numeric_feature_distribution_{var}.png")

    return {
        "text_output": text_output_path,
        "plots_logged": True
    }

# COMMAND ----------

def process_data(df, target_feature, test_size, random_state):
    # Initializing a StringIO object to capture text outputs
    text_output = StringIO()

    # Logging data splitting info
    text_output.write("Data Processing Information:\n\n")
    text_output.write("\nSplitting Data:\n")

    # Separate features and target
    X = df.drop(target_feature, axis=1)
    y = df[target_feature]

    # Split the data into training and testing sets (before resampling)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Logging data shape information
    text_output.write(f"Training Data Shape: {X_train.shape}\n")
    text_output.write(f"Testing Data Shape: {X_test.shape}\n")

    # Apply undersampling only to the training set
    nm = NearMiss()
    X_train_res, y_train_res = nm.fit_resample(X_train, y_train)

    # Logging before and after resampling counts
    text_output.write(f"\nBefore Under-Sampling:\nClass '1' Count: {sum(y_train == 1)}\n")
    text_output.write(f"Class '0' Count: {sum(y_train == 0)}\n")
    text_output.write(f"\nAfter Under-Sampling:\nClass '1' Count: {sum(y_train_res == 1)}\n")
    text_output.write(f"Class '0' Count: {sum(y_train_res == 0)}\n")

    # Convert the training splits into DataFrames for consistency
    train_df = pd.DataFrame(X_train_res, columns=X.columns)
    train_df[target_feature] = y_train_res

    # Testing set remains unchanged
    test_df = pd.DataFrame(X_test, columns=X.columns)
    test_df[target_feature] = y_test

    # Logging final data shapes
    text_output.write(f"\nResampled Training Data Shape: {train_df.shape}\n")
    text_output.write(f"Testing Data Shape (Unchanged): {test_df.shape}\n")

    # Saving the text output to a file
    text_output_path = "process_data_output.txt"
    with open(text_output_path, "w") as f:
        f.write(text_output.getvalue())
    text_output.close()

    # Logging all outputs to MLFlow
    mlflow.log_artifact(text_output_path, artifact_path="Processing")

    # Cleaning up local files
    os.remove(text_output_path)

    return train_df, test_df

# COMMAND ----------

# Main execution
if __name__ == "__main__":
    # Setting up widgets for parameter passing
    dbutils.widgets.text("path", "")
    dbutils.widgets.text("target_feature", "")
    dbutils.widgets.text("test_size", "")
    dbutils.widgets.text("random_state", "")
    dbutils.widgets.text("run_id", "")
    
    # Retrieving parameters
    path = dbutils.widgets.get("path")
    target_feature = dbutils.widgets.get("target_feature")
    test_size = float(dbutils.widgets.get("test_size"))
    random_state = int(dbutils.widgets.get("random_state"))
    run_id = dbutils.widgets.get("run_id")

    # Attaching to the active MLFlow run
    mlflow.start_run(run_id=run_id)

    # Performing EDA and processing
    df = load_data(path)
    df = pd.read_csv(path, encoding="utf-8")
    validate_data(df, target_feature)
    train_df, test_df = process_data(df, target_feature, test_size, random_state)

    # Saving processed training and testing data
    train_df_path = "/dbfs/FileStore/tables/train_data.csv"
    test_df_path = "/dbfs/FileStore/tables/test_data.csv"
    train_df.to_csv(train_df_path, index=False)
    test_df.to_csv(test_df_path, index=False)

    # Logging outputs to MLFlow
    mlflow.log_artifact(train_df_path, artifact_path="EDA/ProcessedData")
    mlflow.log_artifact(test_df_path, artifact_path="EDA/ProcessedData")

    # Exiting with the paths to the processed data
    result = {
        "train_df_path": train_df_path,
        "test_df_path": test_df_path
    }
    dbutils.notebook.exit(json.dumps(result))
