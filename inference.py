"""
Shared inference code for the Gradio (Hugging Face) and Streamlit apps.

It loads the SAME saved artifacts that api.py serves (models/*.joblib) and
applies the SAME input handling as the Flask routes. Nothing is retrained.

Run `python inference.py` to self-test: it prints the input form each app will
build for every model and runs one prediction per model.
"""
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# --------------------------------------------------------------------------
# Locate the models folder (works in the repo, in streamlit_app/, and in a
# Hugging Face Space where models/ sits next to this file).
# --------------------------------------------------------------------------
def _find_models_dir() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in (here / "models", here.parent / "models"):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("Could not find a 'models' folder next to the app.")


MODELS_DIR = _find_models_dir()

# --------------------------------------------------------------------------
# Feature lists (copied from api.py)
# --------------------------------------------------------------------------
CLASSIFICATION_FEATURES = [
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
CLASSIFICATION_NUMERIC = [
    "debt_to_equity_ratio",
    "current_ratio",
    "interest_coverage_ratio",
    "revenue_growth_pct",
    "net_profit_margin_pct",
]

REGRESSION_NUMERIC = [
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
REGRESSION_CATEGORICAL = [
    "store_format",
    "location_type",
    "delivery_available",
    "store_size_category",
]


# --------------------------------------------------------------------------
# Loading (once per process)
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_models() -> dict:
    regression = joblib.load(MODELS_DIR / "retail_sales_regressor.joblib")
    classification = joblib.load(MODELS_DIR / "distress_model_pipeline.joblib")
    cluster_bundle = joblib.load(MODELS_DIR / "ecommerce_clustering_bundle.joblib")
    rules_bundle = joblib.load(MODELS_DIR / "association_rules_bundle.joblib")

    if hasattr(regression, "feature_names_in_"):
        regression_features = list(regression.feature_names_in_)
    else:
        regression_features = REGRESSION_CATEGORICAL + REGRESSION_NUMERIC

    show_names = rules_bundle["show_names"]
    return {
        "regression": regression,
        "regression_features": regression_features,
        "classification": classification,
        "cluster_imputer": cluster_bundle["imputer"],
        "cluster_scaler": cluster_bundle["scaler"],
        "cluster_model": cluster_bundle["model"],
        "cluster_features": list(cluster_bundle["clustering_features"]),
        "cluster_segment_names": cluster_bundle["segment_names"],
        "rules": rules_bundle["rules"],
        "show_names": show_names,
        "name_to_id": {n.lower(): sid for sid, n in show_names.items()},
    }


# --------------------------------------------------------------------------
# Discover dropdown options / default values from a fitted sklearn pipeline
# --------------------------------------------------------------------------
def _flatten(estimator):
    if isinstance(estimator, Pipeline):
        for _, step in estimator.steps:
            yield from _flatten(step)
    else:
        yield estimator


def _native(x):
    return x.item() if hasattr(x, "item") else x


def _is_missing(x) -> bool:
    try:
        return x is None or bool(pd.isna(x))
    except (TypeError, ValueError):
        return False


def discover_pipeline_inputs(pipeline):
    """Return ({column: [allowed categories]}, {column: default value})."""
    categories, defaults = {}, {}
    for est in _flatten(pipeline):
        if not isinstance(est, ColumnTransformer):
            continue
        for _name, trans, cols in est.transformers_:
            if isinstance(trans, str):  # 'drop' / 'passthrough' / 'remainder'
                continue
            if isinstance(cols, str):
                cols = [cols]
            try:
                cols = list(cols)
            except TypeError:
                continue
            if not all(isinstance(c, str) for c in cols):
                continue
            for step in _flatten(trans):
                cats = getattr(step, "categories_", None)
                if cats is not None and len(cats) == len(cols):
                    for col, cat_values in zip(cols, cats):
                        categories[col] = [
                            _native(v) for v in cat_values if not _is_missing(v)
                        ]
                means = getattr(step, "mean_", None)  # e.g. StandardScaler
                if means is not None and len(means) == len(cols):
                    for col, val in zip(cols, means):
                        if not _is_missing(val):
                            defaults.setdefault(col, round(float(val), 4))
                stats = getattr(step, "statistics_", None)
                if stats is not None and len(stats) == len(cols):
                    for col, val in zip(cols, stats):
                        if not _is_missing(val):
                            defaults[col] = _native(val)
    return categories, defaults


def _field(name, kind, default=None, options=None):
    return {"name": name, "kind": kind, "default": default, "options": options}


def _label(name: str) -> str:
    return name.replace("_", " ").capitalize()


@lru_cache(maxsize=1)
def get_specs() -> dict:
    """Describe the input form of each model (used by both UIs)."""
    m = load_models()

    # Regression
    cats, defs = discover_pipeline_inputs(m["regression"])
    reg_fields = []
    for f in m["regression_features"]:
        if f in REGRESSION_NUMERIC:
            reg_fields.append(_field(f, "number", float(defs.get(f, 0.0))))
        elif f in cats and cats[f]:
            default = defs.get(f, cats[f][0])
            reg_fields.append(_field(f, "choice", default, cats[f]))
        else:
            reg_fields.append(_field(f, "text", str(defs.get(f, ""))))

    # Classification
    cats, defs = discover_pipeline_inputs(m["classification"])
    cls_fields = []
    for f in CLASSIFICATION_FEATURES:
        if f in CLASSIFICATION_NUMERIC:
            cls_fields.append(_field(f, "number", float(defs.get(f, 0.0))))
        elif f in cats and cats[f]:
            default = defs.get(f, cats[f][0])
            cls_fields.append(_field(f, "choice", default, cats[f]))
        elif f == "audit_going_concern_flag":
            cls_fields.append(_field(f, "choice", 0, [0, 1]))
        else:
            cls_fields.append(_field(f, "text", ""))

    # Clustering (all numeric; default = training mean)
    means = getattr(m["cluster_scaler"], "mean_", None)
    clu_fields = []
    for i, f in enumerate(m["cluster_features"]):
        default = float(means[i]) if means is not None else 0.0
        clu_fields.append(_field(f, "number", round(default, 4)))

    return {
        "regression": reg_fields,
        "classification": cls_fields,
        "clustering": clu_fields,
        "shows": sorted(m["show_names"].values()),
    }


def _match_option(value, options):
    """UI widgets may hand back '1' for the option 1; map back to the original."""
    if not options:
        return value
    for opt in options:
        if value == opt or str(value) == str(opt):
            return opt
    return value


# --------------------------------------------------------------------------
# Predictions (same logic as the Flask routes in api.py)
# --------------------------------------------------------------------------
def _number(feature, value, allow_missing=False):
    if value is None or (isinstance(value, str) and not value.strip()):
        if allow_missing:
            return float("nan")
        raise ValueError(f"'{feature}' is required and must be numeric.")
    if isinstance(value, bool):
        raise ValueError(f"'{feature}' must be numeric.")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"'{feature}' must be numeric.")


def predict_regression(values: dict) -> dict:
    m = load_models()
    specs = {f["name"]: f for f in get_specs()["regression"]}
    row = {}
    for feature in m["regression_features"]:
        value = values.get(feature)
        if feature in REGRESSION_NUMERIC:
            row[feature] = _number(feature, value, allow_missing=True)
        else:
            if value is None or not str(value).strip():
                raise ValueError(f"'{feature}' must be a non-empty value.")
            value = _match_option(value, specs[feature]["options"])
            row[feature] = value.strip() if isinstance(value, str) else value
    df = pd.DataFrame([row], columns=m["regression_features"])
    log_pred = m["regression"].predict(df)[0]
    # Trained on log1p(daily_sales_revenue_kes) -> expm1 back to KES.
    return {"predicted_daily_sales_revenue_kes": round(float(np.expm1(log_pred)), 2)}


def predict_classification(values: dict) -> dict:
    m = load_models()
    specs = {f["name"]: f for f in get_specs()["classification"]}
    row = {}
    for feature in CLASSIFICATION_FEATURES:
        value = values.get(feature)
        if feature in CLASSIFICATION_NUMERIC:
            row[feature] = _number(feature, value)
        else:
            if value is None or (isinstance(value, str) and not value.strip()):
                raise ValueError(f"'{feature}' is required.")
            row[feature] = _match_option(value, specs[feature]["options"])
    df = pd.DataFrame([row], columns=CLASSIFICATION_FEATURES)
    model = m["classification"]
    result = {"predicted_category": str(model.predict(df)[0])}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(df)[0]
        result["class_probabilities"] = {
            str(label): round(float(p), 4) for label, p in zip(model.classes_, probs)
        }
    return result


def predict_clustering(values: dict) -> dict:
    m = load_models()
    row = [_number(f, values.get(f)) for f in m["cluster_features"]]
    df = pd.DataFrame([row], columns=m["cluster_features"])
    scaled = m["cluster_scaler"].transform(m["cluster_imputer"].transform(df))
    cluster = int(m["cluster_model"].predict(scaled)[0])
    return {
        "predicted_cluster": cluster,
        "segment_name": m["cluster_segment_names"].get(cluster),
    }


def recommend_association(items, top_n=5) -> dict:
    m = load_models()
    if not items:
        raise ValueError("Pick at least one show.")
    if isinstance(top_n, bool) or int(top_n) < 1:
        raise ValueError("'top_n' must be a positive whole number.")
    top_n = int(top_n)

    show_names, name_to_id = m["show_names"], m["name_to_id"]
    watched, unknown = set(), []
    for item in items:
        key = str(item).strip()
        if key.upper() in show_names:
            watched.add(key.upper())
        elif key.lower() in name_to_id:
            watched.add(name_to_id[key.lower()])
        else:
            unknown.append(item)
    if unknown:
        raise ValueError(f"Unknown show(s): {unknown}")

    matching = [r for r in m["rules"] if r["antecedents"] <= watched]
    best = {}
    for rule in matching:
        for show_id in rule["consequents"] - watched:
            cur = best.get(show_id)
            if cur is None or (rule["confidence"], rule["lift"]) > (cur["confidence"], cur["lift"]):
                best[show_id] = rule
    ranked = sorted(
        best.items(),
        key=lambda p: (p[1]["confidence"], p[1]["lift"], p[1]["support"]),
        reverse=True,
    )[:top_n]

    recommendations = [
        {
            "show_id": sid,
            "show_name": show_names[sid],
            "because_you_watched": sorted(show_names[s] for s in rule["antecedents"]),
            "confidence": round(rule["confidence"], 4),
            "lift": round(rule["lift"], 4),
            "support": round(rule["support"], 4),
        }
        for sid, rule in ranked
    ]
    response = {
        "input_items": sorted(show_names[s] for s in watched),
        "recommendations": recommendations,
    }
    if not recommendations:
        response["message"] = (
            "No recommendations found: none of the saved association rules "
            "match these items with a show you have not already watched."
        )
    return response


# --------------------------------------------------------------------------
# Self-test:  python inference.py
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    print("Models folder:", MODELS_DIR)
    specs = get_specs()
    for task in ("regression", "classification", "clustering"):
        print(f"\n=== {task.upper()} form ===")
        for f in specs[task]:
            print(f"  {f['name']:<28} {f['kind']:<7} default={f['default']!r} options={f['options']}")
    print("\n=== RECOMMENDER shows ===")
    print(" ", specs["shows"])

    print("\n=== Test predictions (using each form's default values) ===")
    for task, fn in (
        ("regression", predict_regression),
        ("classification", predict_classification),
        ("clustering", predict_clustering),
    ):
        vals = {f["name"]: f["default"] for f in specs[task]}
        print(f"\n{task}:", json.dumps(fn(vals), indent=2, default=str))
    print("\nrecommender:", json.dumps(recommend_association(specs["shows"][:2]), indent=2))
