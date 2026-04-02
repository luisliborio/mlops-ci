import os
import re
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
import mlflow
import mlflow.sklearn


# MLflow Setup
print("Setting up MLflow...")

#mlflow.set_tracking_uri("http://localhost:5000")
#mlflow.set_experiment("boston-housing")

# mlflow.set_tracking_uri("file:./mlruns") 

# mlflow.set_experiment("boston-housing")

# Check if we are in GitHub Actions
if os.getenv("GITHUB_ACTIONS") == "true":
    # Use a local SQLite database for CI - much more stable
    tracking_uri = "sqlite:///mlflow.db"
else:
    # Use your local server for development
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("boston-housing")

print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")



print("MLflow setup completed!")
# Load Training Data

X_train = pd.read_csv("data/train/X_train.csv")
y_train = pd.read_csv("data/train/y_train.csv")
y_train = y_train.values.ravel()


# Start MLflow Run
print("starting MLflow run for model training...")

with mlflow.start_run():

    # Create Model
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    # Train Model
    model.fit(X_train, y_train)
    print("Model training completed!")

    # Log parameters
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("random_state", 42)

    # Log basic info
    mlflow.log_metric("num_training_samples", len(X_train))

    # Log model artifact
    mlflow.sklearn.log_model(model, "model")

print("MLflow run completed!")

# VERSIONING LOGIC

os.makedirs("models", exist_ok=True)

current_model_file = "models/current_model.txt"

if os.path.exists(current_model_file):

    with open(current_model_file, "r") as f:
        current_model_name = f.read().strip()

    match = re.search(r"model_v(\d+)\.joblib", current_model_name)

    if match:
        current_version = int(match.group(1))
        new_version = current_version + 1
    else:
        new_version = 1
else:
    new_version = 1

# New model name
new_model_name = f"model_v{new_version}.joblib"
model_path = os.path.join("models", new_model_name)

# Save model locally
joblib.dump(model, model_path)
print(f"Model saved at: {model_path}")

# Update active model pointer
with open(current_model_file, "w") as f:
    f.write(new_model_name)

print(f"Active model updated to: {new_model_name}")
