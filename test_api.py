from __future__ import annotations

import os
import sys
from copy import deepcopy

import requests


BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:5000").rstrip("/")

VALID_RECORD = {
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


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_health() -> None:
    response = requests.get(f"{BASE_URL}/health", timeout=30)
    assert_true(response.status_code == 200, f"expected 200, got {response.status_code}")
    body = response.json()
    assert_true(body.get("status") == "healthy", "health status should be healthy")
    assert_true(body.get("model") == "loaded", "model should be loaded")


def test_valid_single_prediction() -> None:
    response = requests.post(f"{BASE_URL}/predict", json=VALID_RECORD, timeout=30)
    assert_true(response.status_code == 200, f"expected 200, got {response.status_code}: {response.text}")
    body = response.json()
    for key in ["prediction", "probability", "label"]:
        assert_true(key in body, f"missing key {key}")
    assert_true(0 <= body["probability"] <= 1, "probability must be within [0, 1]")


def test_valid_batch_prediction() -> None:
    records = []
    for i in range(5):
        record = deepcopy(VALID_RECORD)
        record["delivery_days"] = float(VALID_RECORD["delivery_days"] + i)
        record["total_price"] = float(VALID_RECORD["total_price"] + i * 5)
        record["total_order_value"] = record["total_price"] + record["total_freight"]
        records.append(record)

    response = requests.post(f"{BASE_URL}/predict/batch", json=records, timeout=30)
    assert_true(response.status_code == 200, f"expected 200, got {response.status_code}: {response.text}")
    body = response.json()
    assert_true(body.get("count") == 5, "batch response count should be 5")
    assert_true(len(body.get("predictions", [])) == 5, "expected 5 predictions")


def test_missing_required_field_returns_400() -> None:
    record = deepcopy(VALID_RECORD)
    del record["delivery_days"]
    response = requests.post(f"{BASE_URL}/predict", json=record, timeout=30)
    assert_true(response.status_code == 400, f"expected 400, got {response.status_code}")
    body = response.json()
    assert_true("missing_fields" in body.get("details", {}), "missing field details should be present")


def test_invalid_type_returns_400() -> None:
    record = deepcopy(VALID_RECORD)
    record["total_price"] = "not-a-number"
    response = requests.post(f"{BASE_URL}/predict", json=record, timeout=30)
    assert_true(response.status_code == 400, f"expected 400, got {response.status_code}")
    body = response.json()
    assert_true("total_price" in body.get("details", {}), "invalid type details should mention total_price")


def main() -> int:
    tests = [
        test_health,
        test_valid_single_prediction,
        test_valid_batch_prediction,
        test_missing_required_field_returns_400,
        test_invalid_type_returns_400,
    ]
    print(f"Testing API at {BASE_URL}")
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print("All 5 API tests passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        raise SystemExit(1)
