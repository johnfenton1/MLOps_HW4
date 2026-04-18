from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from flask import Flask, jsonify, request

from common import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURES,
    NON_NEGATIVE_FEATURES,
    NUMERIC_FEATURES,
    PROJECT_DIR,
    example_record,
    load_metadata,
)


MODEL_PATH = Path(os.getenv("MODEL_PATH", PROJECT_DIR / "model" / "model.pkl"))
METADATA_PATH = Path(os.getenv("MODEL_METADATA_PATH", PROJECT_DIR / "model" / "model_metadata.json"))

app = Flask(__name__)

model = None
metadata: dict[str, Any] = {}


def load_artifacts() -> None:
    global model, metadata
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {MODEL_PATH}. Run train_and_serialize.py first."
        )
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Model metadata not found at {METADATA_PATH}. Run train_and_serialize.py first."
        )
    model = joblib.load(MODEL_PATH)
    metadata = load_metadata(METADATA_PATH)


def error_response(details: dict[str, Any], status_code: int = 400):
    return jsonify({"error": "Invalid input", "details": details}), status_code


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_record(record: Any) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if not isinstance(record, dict):
        return None, {"record": "must be a JSON object"}

    details: dict[str, Any] = {}
    missing = [field for field in FEATURES if field not in record]
    if missing:
        details["missing_fields"] = missing

    clean: dict[str, Any] = {}

    for feature in NUMERIC_FEATURES:
        if feature not in record:
            continue
        value = record[feature]
        if not is_number(value):
            details[feature] = "must be a finite number"
            continue
        if feature in NON_NEGATIVE_FEATURES and value < 0:
            details[feature] = "must be non-negative"
            continue
        if feature == "order_hour" and not (0 <= value <= 23):
            details[feature] = "must be between 0 and 23"
            continue
        if feature == "order_dayofweek" and not (0 <= value <= 6):
            details[feature] = "must be between 0 and 6"
            continue
        if feature in BINARY_FEATURES and value not in (0, 1):
            details[feature] = "must be 0 or 1"
            continue
        if feature == "freight_share" and value > 1.5:
            details[feature] = "is unexpectedly high; expected a share near 0 to 1"
            continue
        clean[feature] = value

    allowed_values = metadata.get("categorical_values", {})
    for feature in CATEGORICAL_FEATURES:
        if feature not in record:
            continue
        value = record[feature]
        if not isinstance(value, str) or not value.strip():
            details[feature] = "must be a non-empty string"
            continue
        allowed = allowed_values.get(feature, [])
        if allowed and value not in allowed:
            preview = allowed[:12]
            details[feature] = {
                "message": "unrecognized categorical value",
                "received": value,
                "allowed_examples": preview,
                "allowed_count": len(allowed),
            }
            continue
        clean[feature] = value

    if details:
        return None, details

    return {feature: clean[feature] for feature in FEATURES}, {}


def predict_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(records, columns=FEATURES)
    probabilities = model.predict_proba(frame)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return [
        {
            "prediction": int(pred),
            "probability": round(float(prob), 6),
            "label": "positive" if int(pred) == 1 else "negative",
        }
        for pred, prob in zip(predictions, probabilities)
    ]


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "healthy",
            "model": "loaded" if model is not None else "not_loaded",
            "model_name": metadata.get("model_name"),
            "feature_count": len(FEATURES),
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True)
    clean, details = validate_record(payload)
    if details:
        return error_response(details)
    return jsonify(predict_records([clean])[0])


@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    payload = request.get_json(silent=True)
    if not isinstance(payload, list):
        return error_response({"records": "must be a JSON array of objects"})
    if len(payload) == 0:
        return error_response({"records": "must contain at least one record"})
    if len(payload) > 100:
        return error_response({"records": "batch limit is 100 records"})

    clean_records = []
    batch_errors: dict[str, Any] = {}
    for idx, record in enumerate(payload):
        clean, details = validate_record(record)
        if details:
            batch_errors[str(idx)] = details
        else:
            clean_records.append(clean)

    if batch_errors:
        return error_response(batch_errors)

    return jsonify({"predictions": predict_records(clean_records), "count": len(clean_records)})


@app.route("/", methods=["GET"])
def root():
    return jsonify(
        {
            "service": "Olist customer satisfaction prediction API",
            "health": "/health",
            "single_prediction": "/predict",
            "batch_prediction": "/predict/batch",
            "example": example_record(),
        }
    )


load_artifacts()


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
