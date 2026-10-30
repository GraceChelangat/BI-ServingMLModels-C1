# BI-ServingMLModels-C1


# 2. Financial Distress Classification API
## Overview

This API uses a saved machine learning pipeline to classify a company's
financial condition as Healthy, Watch, or Distressed.

The trained model is loaded from:

`models/distress_model_pipeline.joblib`

The model is loaded when the Flask application starts and is not retrained
for each request.

## Requirements

Install the required packages:

```bash
python -m pip install -r requirements.txt