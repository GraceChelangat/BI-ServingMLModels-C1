from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

# Locate the saved model file.
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "distress_model_pipeline.joblib"

# Loads the trained model once when the API starts.
try:
    classification_model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    raise RuntimeError(
        f"Classification model not found at {MODEL_PATH}. "
        "Place distress_model_pipeline.joblib in the models folder."
    )

# Features expected by the trained classification pipeline.
REQUIRED_FEATURES = [
    "industry_sector",
    "listing_status",
    "credit_rating_band",
    "audit_going_concern_flag",
    "debt_to_equity_ratio",
    "current_ratio",
    "interest_coverage_ratio",
    "revenue_growth_pct",
    "net_profit_margin_pct",
]

NUMERIC_FEATURES = [
    "debt_to_equity_ratio",
    "current_ratio",
    "interest_coverage_ratio",
    "revenue_growth_pct",
    "net_profit_margin_pct",
]


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Financial Distress Classification API is running",
        "endpoint": "/predict/classification",
        "method": "POST",
    }), 200


@app.route("/predict/classification", methods=["POST"])
def predict_classification():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Request body must be a valid JSON object."
        }), 400

    # Check that every required feature is present.
    missing_features = [
        feature for feature in REQUIRED_FEATURES
        if feature not in data
    ]

    if missing_features:
        return jsonify({
            "error": "Missing required features.",
            "missing_features": missing_features,
        }), 400

    # Prepare the input in the same column order used by the model.
    input_data = {feature: data[feature] for feature in REQUIRED_FEATURES}

    # Validate numerical features.
    for feature in NUMERIC_FEATURES:
        try:
            value = input_data[feature]

            if isinstance(value, bool):
                raise ValueError

            input_data[feature] = float(value)

        except (TypeError, ValueError):
            return jsonify({
                "error": f"Feature '{feature}' must be numeric."
            }), 400

    input_df = pd.DataFrame([input_data], columns=REQUIRED_FEATURES)

    try:
        prediction = classification_model.predict(input_df)[0]

        response = {
            "predicted_category": str(prediction)
        }

        # Includes probabilities if the trained model supports them.
        if hasattr(classification_model, "predict_proba"):
            probabilities = classification_model.predict_proba(input_df)[0]
            classes = classification_model.classes_

            response["class_probabilities"] = {
                str(label): round(float(probability), 4)
                for label, probability in zip(classes, probabilities)
            }

        return jsonify(response), 200

    except Exception as error:
        app.logger.exception("Classification prediction failed.")

        return jsonify({
            "error": "Prediction failed. Check the input data and model."
        }), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
