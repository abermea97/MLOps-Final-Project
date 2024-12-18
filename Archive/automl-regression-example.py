# Databricks notebook source
# MAGIC %md
# MAGIC # Use AutoML Cluster 3 for this to run

# COMMAND ----------

# MAGIC %md # AutoML regression example
# MAGIC
# MAGIC ## Requirements
# MAGIC Databricks Runtime for Machine Learning 8.3 or above.

# COMMAND ----------

# MAGIC %md ## California housing dataset
# MAGIC This dataset was derived from the 1990 US census, using one row per census block group. The target variable is the median house value for California districts.

# COMMAND ----------

import sklearn
import pandas as pd

input_pdf = sklearn.datasets.fetch_california_housing(as_frame=True)
display(input_pdf.frame)

# COMMAND ----------

# MAGIC %md ## Train/test split

# COMMAND ----------

from sklearn.model_selection import train_test_split

train_pdf, test_pdf = train_test_split(input_pdf.frame, test_size=0.01, random_state=42)
display(train_pdf)

# COMMAND ----------

# MAGIC %md # Training
# MAGIC The following command starts an AutoML run. You must provide the column that the model should predict in the `target_col` argument.  
# MAGIC When the run completes, you can follow the link to the best trial notebook to examine the training code. This notebook also includes a feature importance plot.

# COMMAND ----------

from databricks import automl
summary = automl.regress(train_pdf, target_col="MedHouseVal", timeout_minutes=30)

# COMMAND ----------

# MAGIC %md The following command displays information about the AutoML output.

# COMMAND ----------

help(summary)

# COMMAND ----------

# MAGIC %md # Next steps
# MAGIC - Explore the notebooks and experiments linked above.
# MAGIC - If the metrics for the best trial notebook look good, skip directly to the inference section.
# MAGIC - If you want to improve on the model generated by the best trial:
# MAGIC   - Go to the notebook with the best trial and clone it.
# MAGIC   - Edit the notebook as necessary to improve the model. For example, you might try different hyperparameters.
# MAGIC   - When you are satisfied with the model, note the URI where the artifact for the trained model is logged. Assign this URI to the `model_uri` variable in Cmd 12.

# COMMAND ----------

# MAGIC %md # Inference
# MAGIC You can use the model trained by AutoML to make predictions on new data. The examples below demonstrate how to make predictions on data in pandas DataFrames, or register the model as a Spark UDF for prediction on Spark DataFrames.

# COMMAND ----------

# MAGIC
# MAGIC %md
# MAGIC ## pandas DataFrame

# COMMAND ----------

model_uri = summary.best_trial.model_path
# model_uri = "<model-uri-from-generated-notebook>"

# COMMAND ----------

import mlflow

# Prepare test dataset
y_test = test_pdf["MedHouseVal"]
X_test = test_pdf.drop("MedHouseVal", axis=1)

# Run inference using the best model
model = mlflow.pyfunc.load_model(model_uri)
predictions = model.predict(X_test)
test_pdf["MedHouseVal_predicted"] = predictions
display(test_pdf)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Spark DataFrame

# COMMAND ----------

# Prepare the test dataset
test_df = spark.createDataFrame(test_pdf)
predict_udf = mlflow.pyfunc.spark_udf(spark, model_uri=model_uri)
display(test_df.withColumn("MedHouseVal_predicted", predict_udf()))

# COMMAND ----------

# MAGIC %md ## Test
# MAGIC Use the final model to make predictions on the holdout test set to estimate how the model would perform in a production setting.

# COMMAND ----------

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Prepare the dataset
y_pred = test_pdf["MedHouseVal_predicted"]
test = pd.DataFrame({"Predicted":y_pred,"Actual":y_test})
test = test.reset_index()
test = test.drop(["index"], axis=1)

# plot graphs
fig= plt.figure(figsize=(16,8))
plt.plot(test[:50])
plt.legend(["Actual", "Predicted"])
sns.jointplot(x="Actual", y="Predicted", data=test, kind="reg");
