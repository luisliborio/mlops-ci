import os
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
import mlflow


# MLflow Setup
# MLflow Setup
print("Setting up MLflow for testing...")

# mlflow.set_tracking_uri("http://localhost:5000")
# mlflow.set_experiment("boston-housing")


# Check if we are in GitHub Actions or have a specific URI set
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(tracking_uri)

print(f"Using tracking URI: {tracking_uri}")

mlflow.set_experiment("boston-housing")


# Load Test Data

X_test = pd.read_csv("data/test/X_test.csv")
y_test = pd.read_csv("data/test/y_test.csv")
y_test = y_test.values.ravel()

MODEL_DIR = "models"
CURRENT_MODEL_FILE = os.path.join(MODEL_DIR, "current_model.txt")

# Read active model name
with open(CURRENT_MODEL_FILE, "r") as f:
    model_name = f.read().strip()

model_path = os.path.join(MODEL_DIR, model_name)

# Load model
model = joblib.load(model_path)


# Start MLflow Run

with mlflow.start_run():

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")

    # Log metrics to MLflow
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)

    # Also log model version
    import re
    match = re.search(r"model_v(\d+)\.joblib", model_name)
    if match:
        mlflow.log_param("model_version", int(match.group(1)))

    # Assertions (quality gate)
    assert mse < 18, "MSE is too high"
    assert mae < 3.0, "MAE is too high"

    print("Model performance test passed!")
