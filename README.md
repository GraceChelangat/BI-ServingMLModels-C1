# Serving ML Models: Student Details and Individual Member Contributions

**Name of the team on GitHub Classroom:** `<your team name>`

---

## Member 1

| Details | Comment |
| ------- | ------- |
| **Student ID** | `170214` |
| **Name** | `Grace Chelangat` |
| **Branch** | [`feature/regression-endpoint`](https://github.com/GraceChelangat/BI-ServingMLModels-C1/tree/feature/regression-endpoint) |

### Contribution

**1. Retail Sales Regression API**

- Trained the Lasso regression model in `regressionC1.ipynb`. The model is a
  scikit-learn pipeline (imputation, scaling, one-hot encoding, Lasso) that predicts
  a store's daily sales revenue in KES.
- Exported the fitted pipeline to `models/retail_sales_regressor.joblib`.
- Added the `POST /predict/regression` endpoint to `api.py`:
  - Loads the saved model once at startup and never retrains it.
  - Validates the request and returns a `400` for a missing or invalid JSON body,
    missing features, non-numeric values in numeric fields, and empty values in
    categorical fields.
  - Accepts `null` for numeric fields because the pipeline imputes missing values.
  - Converts the prediction back to KES with `np.expm1`, because the model was
    trained on `log1p(daily_sales_revenue_kes)`.

**Example request**

```json
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
```

**Example response**

```json
{ "predicted_daily_sales_revenue_kes": 0.0 }
```

Replace the value above with the output from your own test run.

### What I learned

- A saved model must be loaded under the same scikit-learn version it was saved with.
  The classification model was saved with 1.6.1 and my first regression model with
  1.9.0, so one `api.py` could not load both. I re-saved my model under 1.6.1, and
  the version should be pinned in `requirements.txt`.
- When a model is trained on a transformed target (`log1p`), the API has to reverse
  the transformation, otherwise it returns values on the wrong scale.
- Serving a model from disk keeps the API fast and predictable, because nothing is
  retrained on each request.
- Input validation turns confusing server errors into clear `400` responses.

---

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
