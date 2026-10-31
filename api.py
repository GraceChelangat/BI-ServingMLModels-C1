from pathlib import Path
import os
from flask_cors import CORS # type: ignore
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Classification model (Member 2): financial distress
# ---------------------------------------------------------------------------
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

#----------------------------------------------------------------------------
#Regression model (Member 1): retail daily sales revenue (KES)
# ---------------------------------------------------------------------------
REGRESSION_MODEL_PATH = BASE_DIR / "models" / "retail_sales_regressor.joblib"
 
# Loads the trained model once when the API starts (never retrained here).
try:
    regression_model = joblib.load(REGRESSION_MODEL_PATH)
except FileNotFoundError:
    raise RuntimeError(
        f"Regression model not found at {REGRESSION_MODEL_PATH}. "
        "Place retail_sales_regressor.joblib in the models folder."
    )
 
REGRESSION_NUMERIC_FEATURES = [
    "store_age_years",
    "store_size_sqm",
    "num_staff",
    "promotions_active",
    "footfall_count",
    "loyalty_program_members",
    "online_order_count",
    "avg_basket_size_kes",
    "local_events_nearby",
]
 
REGRESSION_CATEGORICAL_FEATURES = [
    "store_format",
    "location_type",
    "delivery_available",
    "store_size_category",
]
 
# Use the exact columns the trained pipeline expects, if it recorded them.
if hasattr(regression_model, "feature_names_in_"):
    REGRESSION_FEATURES = list(regression_model.feature_names_in_)
else:
    REGRESSION_FEATURES = (
        REGRESSION_CATEGORICAL_FEATURES + REGRESSION_NUMERIC_FEATURES
    )
 
 
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "ML model serving API is running",
        "endpoints": {
            "/predict/classification": "POST - financial distress classification",
            "/predict/regression": "POST - retail daily sales revenue (KES)",
        },
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
 
    except Exception:
        app.logger.exception("Classification prediction failed.")
 
        return jsonify({
            "error": "Prediction failed. Check the input data and model."
        }), 500
 
 
@app.route("/predict/regression", methods=["POST"])
def predict_regression():
    data = request.get_json(silent=True)
 
    if not isinstance(data, dict):
        return jsonify({
            "error": "Request body must be a valid JSON object."
        }), 400
 
    missing_features = [f for f in REGRESSION_FEATURES if f not in data]
    if missing_features:
        return jsonify({
            "error": "Missing required features.",
            "missing_features": missing_features,
        }), 400
 
    input_data = {f: data[f] for f in REGRESSION_FEATURES}
 
    for feature in REGRESSION_FEATURES:
        value = input_data[feature]
 
        if feature in REGRESSION_NUMERIC_FEATURES:
            # null is allowed: the pipeline imputes missing values
            # (online_order_count was NaN in the training data).
            if value is None:
                input_data[feature] = float("nan")
                continue
            try:
                if isinstance(value, bool):
                    raise ValueError
                input_data[feature] = float(value)
            except (TypeError, ValueError):
                return jsonify({
                    "error": f"Feature '{feature}' must be numeric."
                }), 400
 
        elif feature in REGRESSION_CATEGORICAL_FEATURES:
            if not isinstance(value, str) or not value.strip():
                return jsonify({
                    "error": f"Feature '{feature}' must be a non-empty string."
                }), 400
            input_data[feature] = value.strip()
 
    input_df = pd.DataFrame([input_data], columns=REGRESSION_FEATURES)
 
    try:
        # The pipeline was trained on log1p(daily_sales_revenue_kes),
        # so np.expm1 converts the prediction back to KES.
        predicted_log_revenue = regression_model.predict(input_df)[0]
        prediction = float(np.expm1(predicted_log_revenue))
 
        return jsonify({
            "predicted_daily_sales_revenue_kes": round(prediction, 2)
        }), 200
 
    except Exception:
        app.logger.exception("Regression prediction failed.")
        return jsonify({
            "error": "Prediction failed. Check the input data and model."
        }), 500
 
 
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)