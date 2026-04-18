from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
PARENT_DIR = PROJECT_DIR.parent

DATASET_CANDIDATES = [
    "olist_cleaned_integrated_dataset.csv",
    "olist_cleaned_integrated_dataset (3).csv",
    "olist_cleaned_integrated_dataset (2).csv",
    "olist_cleaned_integrated_dataset (1).csv",
]

TARGET = "is_positive_review"

NUMERIC_FEATURES = [
    "delivery_days",
    "delivery_vs_estimated",
    "total_price",
    "total_freight",
    "total_order_value",
    "log_total_order_value",
    "n_items",
    "avg_item_price",
    "avg_weight_g",
    "avg_volume_cm3",
    "order_hour",
    "order_dayofweek",
    "approval_hours",
    "freight_share",
    "seller_state_match",
    "is_repeat_customer",
    "is_weekend_purchase",
    "payment_value_total",
]

CATEGORICAL_FEATURES = [
    "product_category",
    "seller_state",
    "payment_type",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

RAW_TO_API_RENAMES = {
    "product_category_name_english": "product_category",
    "payment_type_mode": "payment_type",
}

NON_NEGATIVE_FEATURES = {
    "delivery_days",
    "total_price",
    "total_freight",
    "total_order_value",
    "log_total_order_value",
    "n_items",
    "avg_item_price",
    "avg_weight_g",
    "avg_volume_cm3",
    "approval_hours",
    "freight_share",
    "payment_value_total",
}

BINARY_FEATURES = {
    "seller_state_match",
    "is_repeat_customer",
    "is_weekend_purchase",
}


def find_data_path() -> Path:
    env_path = os.getenv("DATA_PATH")
    if env_path:
        path = Path(env_path)
        if not path.is_absolute():
            path = Path.cwd() / path
        if path.exists():
            return path
        raise FileNotFoundError(f"DATA_PATH does not exist: {path}")

    for base in [PROJECT_DIR, PARENT_DIR, Path.cwd()]:
        for name in DATASET_CANDIDATES:
            path = base / name
            if path.exists():
                return path

    csvs = sorted(PARENT_DIR.glob("*.csv")) + sorted(PROJECT_DIR.glob("*.csv"))
    if csvs:
        return csvs[0]

    raise FileNotFoundError(
        "Could not locate the Olist CSV. Set DATA_PATH to the cleaned integrated dataset."
    )


def load_raw_data(data_path: str | Path | None = None) -> pd.DataFrame:
    path = Path(data_path) if data_path else find_data_path()
    return pd.read_csv(path)


def mode_or_nan(series: pd.Series) -> Any:
    clean = series.dropna()
    if clean.empty:
        return np.nan
    return clean.mode().iloc[0]


def build_order_level_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    for col in [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    required = {
        "order_id",
        "delivery_days",
        "delivery_vs_estimated",
        "price",
        "freight_value",
        "payment_value_total",
        "product_category_name_english",
        "seller_state",
        "payment_type_mode",
        "order_item_id",
        "product_weight_g",
        "product_volume_cm3",
        "order_hour",
        "order_dayofweek",
        "seller_state_match",
        "is_repeat_customer",
        "order_purchase_timestamp",
        "order_approved_at",
        "review_score_mean",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required HW2 columns: {missing}")

    agg = df.groupby("order_id").agg(
        delivery_days=("delivery_days", "first"),
        delivery_vs_estimated=("delivery_vs_estimated", "first"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
        payment_value_total=("payment_value_total", "first"),
        product_category=("product_category_name_english", mode_or_nan),
        seller_state=("seller_state", mode_or_nan),
        payment_type=("payment_type_mode", "first"),
        n_items=("order_item_id", "max"),
        avg_item_price=("price", "mean"),
        avg_weight_g=("product_weight_g", "mean"),
        avg_volume_cm3=("product_volume_cm3", "mean"),
        order_hour=("order_hour", "first"),
        order_dayofweek=("order_dayofweek", "first"),
        seller_state_match=("seller_state_match", "first"),
        is_repeat_customer=("is_repeat_customer", "first"),
        purchase_ts=("order_purchase_timestamp", "first"),
        approved_ts=("order_approved_at", "first"),
        review_score_mean=("review_score_mean", "first"),
    ).reset_index()

    agg[TARGET] = (agg["review_score_mean"] >= 4).astype(int)
    agg["total_order_value"] = agg["total_price"] + agg["total_freight"]
    agg["log_total_order_value"] = np.log1p(agg["total_order_value"].clip(lower=0))
    agg["approval_hours"] = (
        (agg["approved_ts"] - agg["purchase_ts"]).dt.total_seconds() / 3600
    )
    agg["is_weekend_purchase"] = agg["order_dayofweek"].isin([5, 6]).astype(int)
    agg["freight_share"] = agg["total_freight"] / np.where(
        agg["total_order_value"] == 0, np.nan, agg["total_order_value"]
    )

    return agg


def get_feature_target(
    data_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    raw = load_raw_data(data_path)
    agg = build_order_level_dataset(raw)
    return agg[FEATURES].copy(), agg[TARGET].copy(), agg


def categories_from_frame(frame: pd.DataFrame) -> dict[str, list[str]]:
    return {
        feature: sorted(frame[feature].dropna().astype(str).unique().tolist())
        for feature in CATEGORICAL_FEATURES
    }


def save_metadata(path: str | Path, metadata: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_metadata(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def example_record() -> dict[str, Any]:
    return {
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
        "payment_type": "voucher",
    }
