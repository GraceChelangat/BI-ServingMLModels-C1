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

# ---------------- Clustering (K-Means) ----------------
CLUSTER_BUNDLE_PATH = BASE_DIR / "models" / "ecommerce_clustering_bundle.joblib"

try:
    _bundle = joblib.load(CLUSTER_BUNDLE_PATH)
except FileNotFoundError:
    raise RuntimeError(
        f"Clustering bundle not found at {CLUSTER_BUNDLE_PATH}. "
        "Place ecommerce_clustering_bundle.joblib in the models folder."
    )

cluster_imputer = _bundle["imputer"]
cluster_scaler = _bundle["scaler"]
cluster_model = _bundle["model"]
CLUSTER_FEATURES = _bundle["clustering_features"]
CLUSTER_SEGMENT_NAMES = _bundle["segment_names"]


@app.route("/predict/clustering", methods=["POST"])
def predict_clustering():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a valid JSON object."}), 400

    missing_features = [f for f in CLUSTER_FEATURES if f not in data]
    if missing_features:
        return jsonify({
            "error": "Missing required features.",
            "missing_features": missing_features,
        }), 400

    values = []
    for feature in CLUSTER_FEATURES:
        value = data[feature]
        try:
            if isinstance(value, bool):
                raise ValueError
            values.append(float(value))
        except (TypeError, ValueError):
            return jsonify({"error": f"Feature '{feature}' must be numeric."}), 400

    input_df = pd.DataFrame([values], columns=CLUSTER_FEATURES)

    try:
        imputed = cluster_imputer.transform(input_df)
        scaled = cluster_scaler.transform(imputed)
        cluster = int(cluster_model.predict(scaled)[0])
        return jsonify({
            "predicted_cluster": cluster,
            "segment_name": CLUSTER_SEGMENT_NAMES.get(cluster),
        }), 200
    except Exception:
        app.logger.exception("Clustering prediction failed.")
        return jsonify({
            "error": "Prediction failed. Check the input data and model."
        }), 500

# ---------------- Association Rules Recommender ----------------
RULES_BUNDLE_PATH = BASE_DIR / "models" / "association_rules_bundle.joblib"

try:
    _rules_bundle = joblib.load(RULES_BUNDLE_PATH)
except FileNotFoundError:
    raise RuntimeError(
        f"Association rules bundle not found at {RULES_BUNDLE_PATH}. "
        "Place association_rules_bundle.joblib in the models folder."
    )

# Saved rules (frozensets of show IDs) and the show_id -> name lookup.
ASSOCIATION_RULES = _rules_bundle["rules"]
SHOW_NAMES = _rules_bundle["show_names"]
NAME_TO_ID = {name.lower(): sid for sid, name in SHOW_NAMES.items()}


def _resolve_shows(items):
    """Turn show names or show IDs (any case) into show IDs; collect unknown ones."""
    resolved, unknown = set(), []
    for item in items:
        key = str(item).strip()
        if key.upper() in SHOW_NAMES:
            resolved.add(key.upper())
        elif key.lower() in NAME_TO_ID:
            resolved.add(NAME_TO_ID[key.lower()])
        else:
            unknown.append(item)
    return resolved, unknown


@app.route("/recommend/association", methods=["POST"])
def recommend_association():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be a valid JSON object."}), 400

    items = data.get("items")
    if not isinstance(items, list) or len(items) == 0:
        return jsonify({
            "error": "'items' must be a non-empty list of show names or show IDs.",
            "example": {"items": ["Crime City", "The Detective"]},
        }), 400

    top_n = data.get("top_n", 5)
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
        return jsonify({"error": "'top_n' must be a positive whole number."}), 400

    watched, unknown = _resolve_shows(items)
    if unknown:
        return jsonify({
            "error": "Unknown show(s) in 'items'.",
            "unknown_items": unknown,
            "valid_shows": sorted(SHOW_NAMES.values()),
        }), 400

    # A rule matches when its whole antecedent is inside the user's basket.
    matching = [r for r in ASSOCIATION_RULES if r["antecedents"] <= watched]

    # One entry per recommended show not already watched; keep its strongest rule.
    best = {}
    for rule in matching:
        for show_id in rule["consequents"] - watched:
            current = best.get(show_id)
            if current is None or (rule["confidence"], rule["lift"]) > (current["confidence"], current["lift"]):
                best[show_id] = rule

    ranked = sorted(
        best.items(),
        key=lambda pair: (pair[1]["confidence"], pair[1]["lift"], pair[1]["support"]),
        reverse=True,
    )[:top_n]

    recommendations = [
        {
            "show_id": show_id,
            "show_name": SHOW_NAMES[show_id],
            "because_you_watched": sorted(SHOW_NAMES[s] for s in rule["antecedents"]),
            "confidence": round(rule["confidence"], 4),
            "lift": round(rule["lift"], 4),
            "support": round(rule["support"], 4),
        }
        for show_id, rule in ranked
    ]

    response = {
        "input_items": sorted(SHOW_NAMES[s] for s in watched),
        "recommendations": recommendations,
    }
    if not recommendations:
        response["message"] = (
            "No recommendations found: none of the saved association rules "
            "match these items with a show you have not already watched."
        )

    return jsonify(response), 200


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
