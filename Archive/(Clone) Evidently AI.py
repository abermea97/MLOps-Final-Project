# Databricks notebook source
# MAGIC %pip install evidently==0.2.0

# COMMAND ----------

import evidently
import pandas as pd
from sklearn.model_selection import train_test_split

test_df = pd.read_csv("/dbfs/FileStore/tables/test_data.csv")
train_df = pd.read_csv("/dbfs/FileStore/tables/train_data.csv")

import mlflow.pyfunc

# Load the model
model_path = "dbfs:/databricks/mlflow-tracking/1303991687879297/969793f27af7446489a004f6163b7cca/artifacts/model"
model = mlflow.pyfunc.load_model(model_path)

# COMMAND ----------

# Check if the model supports feature importance
if hasattr(model._model_impl, "feature_importances_"):
    feature_importances = model._model_impl.feature_importances_
    feature_names = model._model_impl.feature_names_in_
    
    # Create a DataFrame for visualization
    feature_importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": feature_importances
    }).sort_values(by="Importance", ascending=False)

    print(feature_importance_df)

# COMMAND ----------

from sklearn.model_selection import train_test_split

# Assuming df is your DataFrame and 'target_feature' is your target column
target_feature = "Diabetes_binary"  # Replace with your target column name
test_size = 0.2  # Proportion of data to use for testing
random_state = 42  # For reproducibility

# Separating features (X) and target (y)
X_train = train_df.drop(columns=[target_feature])
y_train = train_df[target_feature]

# COMMAND ----------

#%pip install numpy==1.20
import shap

# Proceed with predictions
predictions = model.predict(prediction_df)

# Assuming you have a training dataset X_train
explainer = shap.Explainer(model.predict, X_train)
shap_values = explainer(X_train)

# Plot the summary for feature importance
shap.summary_plot(shap_values, X_train)

# COMMAND ----------

# Add predictions to both datasets
reference_data["predictions"] = model.predict(reference_data.drop(columns=["target"]))
current_data["predictions"] = model.predict(current_data.drop(columns=["target"]))

# COMMAND ----------

from evidently.report import Report
from evidently.metrics import DataDriftMetric, RegressionPerformanceMetric

# Create a report with selected metrics
report = Report(metrics=[
    DataDriftMetric(),  # Detect data drift in features
    RegressionPerformanceMetric()  # Evaluate model regression performance
])

# Use reference (train) and current (test) datasets
report.run(reference_data=reference_data, current_data=current_data)

# Save the report as HTML
report.save_html("evidently_report.html")
print("Evidently Report created and saved as 'evidently_report.html'")


# COMMAND ----------

from evidently.report import Report
from evidently.metrics import DataDriftMetric, RegressionPerformanceMetric

# Create a report with metrics
report = Report(metrics=[
    DataDriftMetric(),  # Feature drift detection
    RegressionPerformanceMetric()  # Model performance
])

# COMMAND ----------

# Run the report
report.run(reference_data=reference_data, current_data=current_data)

# Save the report as an HTML file
report.save_html("evidently_report.html")
