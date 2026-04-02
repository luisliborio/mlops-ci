import os
import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

MODEL_DIR = "models"
CURRENT_MODEL_FILE = os.path.join(MODEL_DIR, "current_model.txt")


# Load Active Model

with open(CURRENT_MODEL_FILE, "r") as f:
    model_name = f.read().strip()

model_path = os.path.join(MODEL_DIR, model_name)

model = joblib.load(model_path)

print(f"Loaded model: {model_name}")


# Routes


@app.route("/")
def home():
    return "Boston Housing API Running"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Expecting 13 features
        features = np.array(data["features"]).reshape(1, -1)

        prediction = model.predict(features)

        return jsonify({
            "prediction": float(prediction[0]),
            "model_version": model_name
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500



# Run App

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3001)

