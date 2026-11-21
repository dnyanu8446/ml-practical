"""Generate wine class predictions with the selected trained model."""

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.feature_store import ENTITY_COLUMN, TARGET_COLUMN


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict classes for wine feature rows.")
    parser.add_argument("--input", type=Path, required=True, help="CSV with wine features")
    parser.add_argument("--model", type=Path, default=Path("models/best_model.joblib"))
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    args = parser.parse_args()

    features = pd.read_csv(args.input)
    metadata_columns = [
        column for column in (ENTITY_COLUMN, TARGET_COLUMN) if column in features.columns
    ]
    features = features.drop(columns=metadata_columns)
    model = joblib.load(args.model)
    expected_features = list(model.feature_names_in_)
    missing_features = [name for name in expected_features if name not in features.columns]
    extra_features = [name for name in features.columns if name not in expected_features]
    if missing_features or extra_features:
        raise ValueError(
            f"Input feature columns do not match the model; "
            f"missing={missing_features}, extra={extra_features}."
        )
    features = features[expected_features]
    predictions = model.predict(features)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"prediction": predictions}).to_csv(args.output, index=False)
    print(f"Predictions written to {args.output}")


if __name__ == "__main__":
    main()
