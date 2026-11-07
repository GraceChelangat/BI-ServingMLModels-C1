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
| **Student ID** |166567 |
| **Name** |Nicole Wachira |
| **Branch** | [`feature/clustering-endpoint`](https://github.com/GraceChelangat/BI-ServingMLModels-C1/tree/feature/clustering-endpoint) |

### Contribution

**1. E-commerce Customer Clustering API**

- Used the K-Means model our group trained in the clustering lab (`ecommerce_customer_clustering.ipynb`). It was saved as a bundle (`models/ecommerce_clustering_bundle.joblib`) holding the median imputer, StandardScaler, K-Means model (5 clusters), the feature list and the segment names.
- Added the `POST /predict/clustering` endpoint to the shared `api.py`:
  - Loads the saved bundle once at startup, using a path built from `BASE_DIR`, and never retrains it.
  - Applies the same steps as training (impute, scale, predict) with the four features in training order.
  - Returns the cluster number and its segment name.
  - Validates the request and returns a 400 for a non-JSON body, missing features and non-numeric values.
- Fixed the local environment: the bundle was saved with scikit-learn 1.6.1, which has no package for Python 3.14, so I created a Python 3.12 virtual environment and installed the team's pinned `requirements.txt`.
- Tested with Postman and curl: an active customer returned cluster 1 and a dormant customer returned cluster 2, matching the notebook, and the error cases returned 400.

**Endpoint**

`POST /predict/clustering`

**Request**

```json
{
  "recency_days": 5,
  "frequency_per_year": 28,
  "avg_order_value_kes": 9500,
  "num_categories_purchased": 6
}
```

**Response**

```json
{
  "predicted_cluster": 1,
  "segment_name": "High-Frequency Big Spenders"
}
```

**Error response (missing field, 400)**

```json
{
  "error": "Missing required features.",
  "missing_features": ["frequency_per_year", "avg_order_value_kes", "num_categories_purchased"]
}
```

**What I learned**

- New data must go through exactly the same preprocessing and feature order as the training data. Otherwise the model still returns a cluster number, but a wrong one, with no error.
- A saved model must be loaded under the same scikit-learn version it was saved with. Our bundle was saved with 1.6.1. A newer version loaded the file but crashed at prediction time, so I used a matching environment.
- Models are loaded once at startup and never retrained inside the API. This is faster and keeps predictions consistent.
- Clear error messages (400) and documentation matter, because the API is how other people use the model.
- A 405 Method Not Allowed means the route exists but was called with the wrong HTTP method (opening a POST route in the browser sends GET).

---


## Member 4

| Details | Comment |
| ------- | ------- |
| **Student ID** | `160819` |
| **Name** | `Alfred Njaku` |
| **Branch** | [`feature/association-rules-endpoint`](https://github.com/GraceChelangat/BI-ServingMLModels-C1/tree/feature/association-rules-endpoint) |

### Contribution

**4. Streaming Series Association Rules Recommender API**

- Used the 21 cleaned association rules our group saved in the Association Rule Mining lab (`AssociationRuleMining-C1/outputs/clean_association_rules.pkl`, mined with min support 0.02, min confidence 0.30, min lift 2.0) and the show lookup table (`outputs/show_lookup.csv`).
- Wrote `scripts/save_association_bundle.py`, which loads those saved rules (without re-mining them) and saves them with joblib as `models/association_rules_bundle.joblib`, the same format the other members use. The bundle holds the rules, the `show_id` → show name lookup and the mining thresholds. The rules are stored as plain Python objects (frozensets and floats, not a DataFrame), so loading them does not depend on the pandas version on the server.
- Added the `POST /recommend/association` endpoint to the shared `api.py`:
  - Loads the saved bundle once at startup with `joblib.load`, using a path built from `BASE_DIR`, and never re-mines the rules.
  - Accepts the shows a viewer has watched, as show names or show IDs, in any letter case.
  - A rule matches when its whole antecedent is contained in the viewer's basket.
  - Removes shows the viewer has already watched, keeps one entry per recommended show (its strongest rule), and ranks by confidence, then lift, then support.
  - Returns an empty list with a clear message when no rule matches.
  - Returns a 400 for a non-JSON body, a missing or empty `items` list, unknown shows (listing the valid ones) and an invalid `top_n`.
- Added `tests/test_association.py` (10 pytest tests) covering a match, name/ID equivalence, `top_n`, no match, unknown shows and bad requests.

**Endpoint**

`POST /recommend/association`

| Field | Type | Required | Description |
| ----- | ---- | -------- | ----------- |
| `items` | list of strings | Yes | Shows already watched, by name (`"Crime City"`) or ID (`"S01"`) |
| `top_n` | integer | No (default 5) | Maximum number of recommendations |

**Request**

```json
{
  "items": ["Galactic Wars", "Time Anomaly"]
}
```

**Response (200)**

```json
{
  "input_items": ["Galactic Wars", "Time Anomaly"],
  "recommendations": [
    {
      "show_id": "S06",
      "show_name": "Space Frontier",
      "because_you_watched": ["Galactic Wars", "Time Anomaly"],
      "confidence": 0.7681,
      "lift": 4.844,
      "support": 0.0303
    }
  ]
}
```

**No matching rule (200)**

```json
{
  "input_items": ["Toon Town"],
  "recommendations": [],
  "message": "No recommendations found: none of the saved association rules match these items with a show you have not already watched."
}
```

**Unknown show (400)**

```json
{
  "error": "Unknown show(s) in 'items'.",
  "unknown_items": ["Not A Real Show"],
  "valid_shows": ["Cooking Masters", "Crime City", "..."]
}
```

**cURL**

```bash
curl -X POST http://127.0.0.1:5000/recommend/association \
  -H "Content-Type: application/json" \
  -d '{"items": ["Crime City", "The Detective"]}'
```

**What I learned**

- An association rule recommender has no "model" to predict with; the saved rules *are* the model, so serving it means loading the rules and matching them, not retraining.
- A rule only applies when its whole antecedent is in the basket, and shows the user has already watched should not be recommended back.
- Several rules can point to the same show, so results need de-duplicating before they reach the user.
- "No recommendation" is a valid answer and should return 200 with a message, while bad input should return 400.
- A saved file is tied to the library versions it was saved with. Keeping the bundle to plain Python objects avoids the pandas and scikit-learn version problems other members hit.

---

## Member 5

| Details        | Comment               |
|----------------|-----------------------|
| **Student ID** | 169302                |
| **Name**       | Anastacia Muya        |
| **Branch**     | feature/docker-nginx  https://github.com/GraceChelangat/BI-ServingMLModels-C1/tree/feature/docker-nginx|



**Contribution**
5. Docker, Gunicorn and Nginx Deployment

- Wrote the Dockerfile on a python:3.11-slim base. It installs the packages from requirements.txt, copies the project (including the models/ folder) into the image, and starts the API with Gunicorn: `gunicorn --bind 0.0.0.0:5000 --workers 2 --timeout 120 api:app`. The Flask development server is not used.
- Wrote nginx/nginx.conf, a reverse proxy that forwards all requests to the API container (`proxy_pass http://api:5000`) and passes on the Host and client IP headers.
- Wrote docker-compose.yml with two services. `api` is built from the Dockerfile and only exposed inside the Docker network. `nginx` uses the nginx:stable image and is published on port 8080. The API can only be reached through Nginx.
- Added a .dockerignore that leaves out .git, caches and virtual environments but keeps models/, so the saved .joblib files are inside the container.
- Did not change api.py. The same api.py runs locally, in Docker and on Render.
- Verified the build: the image built successfully, and the startup logs show `Starting gunicorn`, `Listening at: http://0.0.0.0:5000`, `Using worker: sync` and two workers booting. Both containers were created and started.
- Verified the models are inside the container: all model files loaded at startup without the RuntimeError that api.py raises when a file is missing, and I loaded the clustering and classification files from inside the running container to read their feature names and categories.
- Tested through the Nginx port (8080), not the Flask server. Every response carries `Server: nginx/1.30.6`. Endpoints tested: clustering, recommender, classification and regression.
- Added a simple HTML/CSS/JavaScript demo page (frontend/index.html) served by the same Nginx container at http://localhost:8080/demo/. It calls the API with relative paths (/predict/clustering and /recommend/association) and shows an error message when an input is missing. api.py was not changed.
**How to run**

Prerequisite: Docker Desktop installed and running.
```
docker compose up --build
```
Check both containers are up:
```
docker compose ps
```
Test through Nginx:
```
curl -i http://localhost:8080/
```
Example response:
```
HTTP/1.1 200 OK
Server: nginx/1.30.6
Content-Type: application/json

{"endpoint":"/predict/classification","message":"Financial Distress Classification API is running","method":"POST"}
```
**Example model request through Nginx:**
```
curl -i -X POST http://localhost:8080/predict/clustering \
  -H "Content-Type: application/json" \
  -d '{"recency_days": 300, "frequency_per_year": 1, "avg_order_value_kes": 500, "num_categories_purchased": 1}'
```
**Example response (200):**
```
{"predicted_cluster": 2, "segment_name": "Dormant Low-Value Accounts"}
```
Stop everything with: 
```
docker compose down
```

Files added

| File | Purpose |
|---|---|
| Dockerfile | Builds the API image and starts Gunicorn |
| docker-compose.yml | Runs the api and nginx containers together |
| nginx/nginx.conf | Reverse proxy settings |
| .dockerignore | Keeps unwanted files out of the image but keeps models/ |
| frontend/index.html | Demo page for the API, with missing-input error handling   |

Troubleshooting

- 502 Bad Gateway: the API container crashed. Run `docker compose logs api` to see the error.
- Port 8080 already in use: change "8080:80" to "8081:80" in docker-compose.yml and use port 8081.
- "Cannot connect to the Docker daemon": Docker Desktop is not running. Open it and wait until it says running.
- Model not found in the container: the model file is missing from models/ on GitHub, or .dockerignore excludes it.
- If the first build is slow, that is normal. Later builds use the cache.

What I learned

- Gunicorn is a production server, while `app.run()` is only for development. Gunicorn finds the app through the `module:variable` name (`api:app`).
- Nginx is a reverse proxy: clients talk to Nginx, and Nginx talks to the API. Inside Docker Compose, containers reach each other by service name, which is why it forwards to `http://api:5000`.
- `expose` keeps a port private to the Docker network, while `ports` publishes it to the host. Only Nginx is published.
- Saved model files must be inside the image, so .dockerignore must not exclude models/, and the code must build paths from BASE_DIR instead of using absolute paths from a laptop.
- A saved model is tied to the library versions it was created with, so requirements.txt needs compatible versions.
- A running container is only proof if it answers real requests, so I tested through port 8080 and checked the logs and the `Server: nginx` header.
- Reading a trained pipeline shows which inputs it accepts. For example, credit_rating_band only accepts Investment Grade, Speculative or High Risk, so the exact values matter when writing test requests.

Requirements
Install the required packages:
```
python -m pip install -r requirements.txt
```