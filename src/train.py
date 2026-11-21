"""Compare wine classifiers, track runs in MLflow, and save the best model."""

import argparse
import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from src.feature_store import load_feature_table


def build_models(model_params: dict, random_state: int) -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=model_params["logistic_regression"]["max_iter"],
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=model_params["random_forest"]["n_estimators"],
                        random_state=random_state,
                    ),
                )
            ]
        ),
        "svc": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", SVC(C=model_params["svc"]["c"])),
            ]
        ),
    }


def train_and_track(
    features_path: Path,
    schema_path: Path,
    params_path: Path,
    model_path: Path,
    metrics_path: Path,
) -> dict:
    params = yaml.safe_load(params_path.read_text(encoding="utf-8"))
    table, schema = load_feature_table(features_path, schema_path)
    x = table[schema["features"]]
    y = table[schema["target"]]

    split_params = params["split"]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=split_params["test_size"],
        random_state=split_params["random_state"],
        stratify=y,
    )
    folds = params["cv"]["folds"]
    smallest_class = int(y_train.value_counts().min())
    if folds < 2 or folds > smallest_class:
        raise ValueError(
            f"cv.folds must be between 2 and the smallest training class "
            f"count ({smallest_class}); got {folds}."
        )

    cv = StratifiedKFold(
        n_splits=folds, shuffle=True, random_state=split_params["random_state"]
    )
    models = build_models(params["models"], split_params["random_state"])

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("wine-classification")

    results = {}
    fitted_models = {}
    for model_name, model in models.items():
        cv_scores = cross_val_score(model, x_train, y_train, cv=cv, scoring="accuracy")
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        result = {
            "cv_accuracy_mean": float(cv_scores.mean()),
            "cv_accuracy_std": float(cv_scores.std()),
            "test_accuracy": float(accuracy_score(y_test, predictions)),
            "test_weighted_f1": float(f1_score(y_test, predictions, average="weighted")),
        }
        results[model_name] = result
        fitted_models[model_name] = model

        with mlflow.start_run(run_name=model_name):
            mlflow.log_param("model", model_name)
            mlflow.log_param("cv_folds", folds)
            mlflow.log_param("test_size", split_params["test_size"])
            mlflow.log_param("random_state", split_params["random_state"])
            for name, value in params["models"][model_name].items():
                mlflow.log_param(name, value)
            mlflow.log_metrics(result)
            mlflow.set_tag("feature_store", schema["name"])
            trusted_types = (
                ["sklearn.tree._tree.Tree"] if model_name == "random_forest" else None
            )
            mlflow.sklearn.log_model(
                model, name="model", skops_trusted_types=trusted_types
            )

    best_model_name = max(
        results, key=lambda name: (results[name]["cv_accuracy_mean"], name)
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted_models[best_model_name], model_path)

    summary = {
        "selected_model": best_model_name,
        "selection_metric": "cv_accuracy_mean",
        "models": results,
        "test_rows": len(y_test),
        "train_rows": len(y_train),
        "feature_count": len(schema["features"]),
    }
    metrics_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and compare wine classifiers.")
    parser.add_argument(
        "--features", type=Path, default=Path("data/processed/wine_features.csv")
    )
    parser.add_argument("--schema", type=Path, default=Path("feature_store/schema.json"))
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    parser.add_argument("--model", type=Path, default=Path("models/best_model.joblib"))
    parser.add_argument("--metrics", type=Path, default=Path("metrics/summary.json"))
    args = parser.parse_args()

    summary = train_and_track(
        args.features, args.schema, args.params, args.model, args.metrics
    )
    print(f"Selected model: {summary['selected_model']}")
    print(f"Metrics written to {args.metrics}")
    print(f"Model written to {args.model}")


if __name__ == "__main__":
    main()
