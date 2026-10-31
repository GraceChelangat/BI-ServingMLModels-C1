# Serving ML Models: Student Details and Individual Member Contributions

**Name of the team on GitHub Classroom:** `C1`

---

## Member 1

| Details | Comment |
| ------- | ------- |
| **Student ID** | `170214` |
| **Name** | `Grace Chelangat` |
| **Branch** | [`feature/regression-endpoint`](https://github.com/GraceChelangat/BI-ServingMLModels-C1/tree/feature/regression-endpoint) |

### Contribution

**1. Retail Sales Regression API**

Contribution

1. Retail Sales Regression API

Trained a Lasso regression pipeline (imputation, scaling, one-hot encoding, Lasso) in regressionC1.ipynb to predict a store's daily sales revenue in KES, and exported it to models/retail_sales_regressor.joblib.
Added the POST /predict/regression endpoint to the shared api.py:
Loads the saved model once at startup and never retrains it.
Validates the request and returns a 400 for a missing or invalid JSON body, missing features, non-numeric values in numeric fields, and empty values in categorical fields.
Accepts null for numeric fields because the pipeline imputes missing values.
Converts the prediction back to KES with np.expm1, because the model was trained on log1p(daily_sales_revenue_kes).
Deployed the combined API (classification and regression) to Render with Gunicorn. requirements.txt pins scikit-learn==1.6.1, and the Python version is pinned. The live service responds on / and lists both endpoints.
Debugged deployment failures from the Render logs: a syntax error from a clipped line, a leftover requests import from the lab example, a missing flask-cors dependency, a malformed requirements.txt, and Render defaulting to Python 3.14.
Endpoint

POST /predict/regression

Request

json
{
  "store_format": "Supermarket",
  "location_type": "Mall",
  "delivery_available": "Yes",
  "store_size_category": "Medium",
  "store_age_years": 5.0,
  "store_size_sqm": 450.0,
  "num_staff": 12,
  "promotions_active": 1,
  "footfall_count": 800.0,
  "loyalty_program_members": 150,
  "online_order_count": 40.0,
  "avg_basket_size_kes": 1200.0,
  "local_events_nearby": 0
}

Response
{
    "endpoints": {
        "/predict/classification": "POST - financial distress classification",
        "/predict/regression": "POST - retail daily sales revenue (KES)"
    },
    "message": "ML model serving API is running"
}

What I learned
A saved model must be loaded under the same scikit-learn version it was saved with. My model was saved with 1.9.0 and the classification model with 1.6.1, so one api.py could not load both. I re-saved my model under 1.6.1 and pinned the version in requirements.txt.
If the target was transformed during training (log1p), the API has to reverse it, otherwise predictions come back on the wrong scale.
Deployment errors are best read from the server logs, because the API's own error messages are deliberately generic.
A 405 Method Not Allowed means the route exists but was called with the wrong HTTP method.

## Member 2

| Details | Comment |
| ------- | ------- |
| **Student ID** | |
| **Name** | |

### Contribution

**2. Financial Distress Classification API**

#### Overview

This API uses a saved machine learning pipeline to classify a company's
financial condition as Healthy, Watch, or Distressed.

The trained model is loaded from:

`models/distress_model_pipeline.joblib`

---

## Member 3

| Details | Comment |
| ------- | ------- |
| **Student ID** | |
| **Name** | |

### Contribution

*What part of the lab did you personally contribute to (provide a link to the
branch(es)), and what did you learn from it?*

---

## Member 4

| Details | Comment |
| ------- | ------- |
| **Student ID** | |
| **Name** | |

### Contribution

*What part of the lab did you personally contribute to (provide a link to the
branch(es)), and what did you learn from it?*

---

## Member 5

| Details | Comment |
| ------- | ------- |
| **Student ID** | |
| **Name** | |

### Contribution

*What part of the lab did you personally contribute to (provide a link to the
branch(es)), and what did you learn from it?*


## Requirements

Install the required packages:

```bash
python -m pip install -r requirements.txt
