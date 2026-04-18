from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    PROJECT_DIR,
    TARGET,
    categories_from_frame,
    find_data_path,
    get_feature_target,
    save_metadata,
)


REGISTERED_MODEL_NAME = "olist-satisfaction-model"


def build_preprocessor() -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def metrics_for(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
    }


def train_model(data_path: str | None = None, log_mlflow: bool = True) -> dict[str, object]:
    resolved_data_path = Path(data_path) if data_path else find_data_path()
    X, y, agg = get_feature_target(resolved_data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    model_dir = PROJECT_DIR / "model"
    model_dir.mkdir(exist_ok=True)

    models = {
        "logistic_regression_baseline": Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                ("model", LogisticRegression(max_iter=2000)),
            ]
        ),
        "tuned_random_forest_hw2": Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=20,
                        min_samples_split=5,
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }

    results: dict[str, dict[str, float]] = {}
    fitted_models: dict[str, Pipeline] = {}

    if log_mlflow:
        mlflow.set_tracking_uri(f"file:{(PROJECT_DIR / 'mlruns').as_posix()}")
        mlflow.set_experiment("olist-satisfaction")
        mlflow_client = MlflowClient()
    else:
        mlflow_client = None

    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        fitted_models[name] = pipe
        results[name] = metrics_for(pipe, X_test, y_test)

        if log_mlflow:
            with mlflow.start_run(run_name=name):
                run_id = mlflow.active_run().info.run_id
                clf = pipe.named_steps["model"]
                mlflow.log_param("model_type", clf.__class__.__name__)
                for param, value in clf.get_params().items():
                    if param in {"n_estimators", "max_depth", "min_samples_split", "max_iter", "class_weight"}:
                        mlflow.log_param(param, value)
                mlflow.log_param("feature_count", len(FEATURES))
                mlflow.log_param("target", TARGET)
                mlflow.log_metrics(results[name])

                artifact_dir = PROJECT_DIR / "model" / "mlflow_artifacts"
                artifact_dir.mkdir(parents=True, exist_ok=True)

                if name == "tuned_random_forest_hw2":
                    mlflow.sklearn.log_model(
                        sk_model=pipe,
                        artifact_path="model",
                        input_example=X_test.head(3),
                        registered_model_name=REGISTERED_MODEL_NAME,
                    )

                    if mlflow_client is not None:
                        versions = mlflow_client.search_model_versions(
                            f"name='{REGISTERED_MODEL_NAME}'"
                        )
                        run_versions = [v for v in versions if v.run_id == run_id]
                        latest_version = max(
                            run_versions or versions,
                            key=lambda v: int(v.version),
                        )
                        try:
                            mlflow_client.transition_model_version_stage(
                                name=REGISTERED_MODEL_NAME,
                                version=latest_version.version,
                                stage="Production",
                                archive_existing_versions=True,
                            )
                            mlflow.set_tag("registered_model_stage", "Production")
                        except Exception as exc:
                            mlflow_client.set_registered_model_alias(
                                REGISTERED_MODEL_NAME,
                                "production",
                                latest_version.version,
                            )
                            mlflow.set_tag(
                                "registered_model_stage",
                                f"Production stage unsupported; set alias production instead: {exc}",
                            )
                        print(
                            "Registered MLflow model "
                            f"{REGISTERED_MODEL_NAME} version {latest_version.version}"
                        )
                else:
                    run_model_path = artifact_dir / f"{name}.pkl"
                    joblib.dump(pipe, run_model_path)
                    mlflow.log_artifact(str(run_model_path), artifact_path="model")

                metrics_path = artifact_dir / f"{name}_metrics.csv"
                pd.DataFrame([results[name]]).to_csv(metrics_path, index=False)
                mlflow.log_artifact(str(metrics_path), artifact_path="metrics")

    best_name = "tuned_random_forest_hw2"
    best_model = fitted_models[best_name]
    model_path = model_dir / "model.pkl"
    joblib.dump(best_model, model_path)

    metadata = {
        "dataset_filename": resolved_data_path.name,
        "dataset_path": str(resolved_data_path),
        "model_name": best_name,
        "model_type": "RandomForestClassifier",
        "target": TARGET,
        "positive_label": "review_score_mean >= 4",
        "negative_label": "review_score_mean <= 3",
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "categorical_values": categories_from_frame(X),
        "test_metrics": results[best_name],
        "comparison_metrics": results,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "total_order_rows": int(len(X)),
        "class_balance_positive_rate": float(y.mean()),
        "hyperparameters": {
            "n_estimators": 200,
            "max_depth": 20,
            "min_samples_split": 5,
            "random_state": 42,
            "n_jobs": 1,
        },
    }
    save_metadata(model_dir / "model_metadata.json", metadata)

    print(f"Detected dataset: {resolved_data_path.name}")
    print(f"Saved model: {model_path}")
    print("API feature list:")
    for feature in FEATURES:
        print(f"- {feature}")
    print("Test metrics:")
    for metric, value in results[best_name].items():
        print(f"{metric}: {value:.4f}")

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--skip-mlflow", action="store_true")
    args = parser.parse_args()
    train_model(data_path=args.data_path, log_mlflow=not args.skip_mlflow)


if __name__ == "__main__":
    main()
