# Olist Customer Satisfaction MLOps Pipeline

## Project Overview

This project productionizes the Olist customer satisfaction model developed in HW2. The deployed model predicts whether an order will receive a positive review, where `review_score_mean >= 4` is positive and scores 1-3 are negative.

The production model is the tuned HW2 Random Forest pipeline with the same order-level aggregation, engineered features, preprocessing, and target definition used in the original notebook.

## Live URL

Render deployment URL: `https://your-render-service.onrender.com`

Replace this placeholder after deploying the Dockerized API to Render.

## Model Information

- Business problem: proactive customer satisfaction prediction for Olist orders.
- Target: `is_positive_review = 1` when `review_score_mean >= 4`, otherwise `0`.
- Production model: `RandomForestClassifier`.
- HW2-derived production hyperparameters: `n_estimators=200`, `max_depth=20`, `min_samples_split=5`, `random_state=42`, `n_jobs=1`.
- Compatibility note: the exact unconstrained 400-tree HW2 Random Forest serialized to more than 700 MB locally, so the deployed version keeps the same HW2 pipeline and Random Forest structure but caps depth and tree count for Render-ready serving.
- Preprocessing: median imputation and standard scaling for numeric features; most-frequent imputation and one-hot encoding for categorical features.
- Serialized artifact: `model/model.pkl`.
- Experiment tracking: `train_and_serialize.py` creates an MLflow experiment named `olist-satisfaction` and logs baseline Logistic Regression and tuned Random Forest runs.

Known limitations:

- The model is proactive: it uses order, payment, product, and delivery features, not the written review text.
- The cleaned local modeling dataset does not include `review_comment_message`, so the foundation-model notebook documents that limitation and is ready to run if a review-text Olist file is supplied.
- Categorical values are validated against the training data. New seller states, payment types, or product categories require either validation updates or model retraining.
- The model was trained on historical Olist data; drift in logistics, freight, product mix, or customer expectations can reduce performance.

## API Documentation

### `GET /health`

Returns model and service status.

Example response:

```json
{
  "status": "healthy",
  "model": "loaded",
  "model_name": "tuned_random_forest_hw2",
  "feature_count": 21
}
```

### `POST /predict`

Returns one prediction for one order-level feature record.

Example request:

```json
{
  "delivery_days": 8.0,
  "delivery_vs_estimated": -8.0,
  "total_price": 29.99,
  "total_freight": 8.72,
  "total_order_value": 38.71,
  "log_total_order_value": 3.682862,
  "n_items": 1.0,
  "avg_item_price": 29.99,
  "avg_weight_g": 500.0,
  "avg_volume_cm3": 1976.0,
  "order_hour": 10,
  "order_dayofweek": 0,
  "approval_hours": 0.178333,
  "freight_share": 0.225265,
  "seller_state_match": 1,
  "is_repeat_customer": 1,
  "is_weekend_purchase": 0,
  "payment_value_total": 38.71,
  "product_category": "housewares",
  "seller_state": "SP",
  "payment_type": "voucher"
}
```

Example response:

```json
{
  "prediction": 1,
  "probability": 0.86,
  "label": "positive"
}
```

### `POST /predict/batch`

Accepts a JSON array of 1 to 100 records using the same schema as `/predict`.

Example response:

```json
{
  "predictions": [
    {"prediction": 1, "probability": 0.86, "label": "positive"},
    {"prediction": 0, "probability": 0.31, "label": "negative"}
  ],
  "count": 2
}
```

### Error Format

Invalid requests return HTTP 400 with helpful details.

```json
{
  "error": "Invalid input",
  "details": {
    "missing_fields": ["delivery_days"]
  }
}
```

## Input Schema

| Feature | Type | Valid values / notes |
|---|---:|---|
| `delivery_days` | number | Non-negative delivery duration in days |
| `delivery_vs_estimated` | number | Actual delivery days minus estimated delivery days; negative means early |
| `total_price` | number | Non-negative total item price for the order |
| `total_freight` | number | Non-negative total freight value |
| `total_order_value` | number | `total_price + total_freight`; non-negative |
| `log_total_order_value` | number | `log1p(total_order_value)`; non-negative |
| `n_items` | number | Non-negative item count from order item ids |
| `avg_item_price` | number | Non-negative average item price |
| `avg_weight_g` | number | Non-negative average product weight in grams |
| `avg_volume_cm3` | number | Non-negative average product volume |
| `order_hour` | number | 0 through 23 |
| `order_dayofweek` | number | 0 through 6 |
| `approval_hours` | number | Non-negative hours from purchase to approval |
| `freight_share` | number | Freight divided by total order value |
| `seller_state_match` | number | 0 or 1 |
| `is_repeat_customer` | number | 0 or 1 |
| `is_weekend_purchase` | number | 0 or 1 |
| `payment_value_total` | number | Non-negative total payment value |
| `product_category` | string | Must match a product category observed in training |
| `seller_state` | string | Must match a seller state observed in training |
| `payment_type` | string | One of the payment types observed in training |

The exact categorical value lists are saved in `model/model_metadata.json` after training.

## Local Setup Without Docker

```powershell
cd "C:\Users\jfent\Downloads\New folder\MLOps\hw4-mlops"
python -m pip install -r requirements.txt
python train_and_serialize.py
python app.py
```

In a second terminal:

```powershell
cd "C:\Users\jfent\Downloads\New folder\MLOps\hw4-mlops"
python test_api.py
```

To point tests at a deployed API:

```powershell
$env:API_BASE_URL="https://your-render-service.onrender.com"
python test_api.py
```

## Local Setup With Docker

```powershell
cd "C:\Users\jfent\Downloads\New folder\MLOps\hw4-mlops"
docker build -t hw4-api .
docker run --rm -p 5000:5000 hw4-api
```

In a second terminal:

```powershell
python test_api.py
```

## MLflow

Training logs two runs to the `olist-satisfaction` experiment:

```powershell
python train_and_serialize.py
mlflow ui --backend-store-uri .\mlruns
```

Open the MLflow UI, capture the experiment screenshot, register the best Random Forest model, transition it to Production, and capture the model registry screenshot for Blackboard.

## Project Files

- `part1_foundation_model.ipynb`: foundation sentiment model comparison.
- `part5_monitoring.ipynb`: six-month drift and performance monitoring simulation.
- `train_and_serialize.py`: reproducible training, MLflow logging, and model serialization.
- `common.py`: shared HW2 feature engineering and schema.
- `app.py`: Flask API.
- `test_api.py`: five required API tests.
- `Dockerfile`: Render-compatible container image.
