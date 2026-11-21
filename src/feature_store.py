"""Build and validate the project's local offline feature store."""

import argparse
import json
from pathlib import Path

import pandas as pd


TARGET_COLUMN = "target"
ENTITY_COLUMN = "wine_id"


def build_feature_table(
    input_path: Path, output_path: Path, schema_path: Path
) -> None:
    data = pd.read_csv(input_path)
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Input data must contain a '{TARGET_COLUMN}' column.")
    if data.empty:
        raise ValueError("Input data contains no rows.")
    if data.columns.duplicated().any():
        raise ValueError("Input data contains duplicate column names.")

    feature_columns = [column for column in data.columns if column != TARGET_COLUMN]
    if not all(pd.api.types.is_numeric_dtype(data[column]) for column in data.columns):
        raise ValueError("All feature and target columns must be numeric.")
    if data.isna().any().any():
        raise ValueError("Input data contains missing values.")

    table = data.copy()
    table.insert(0, ENTITY_COLUMN, range(len(table)))

    schema = {
        "name": "wine_offline_features",
        "version": 1,
        "entity": ENTITY_COLUMN,
        "features": feature_columns,
        "target": TARGET_COLUMN,
        "row_count": len(table),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    schema_path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")


def load_feature_table(features_path: Path, schema_path: Path) -> tuple[pd.DataFrame, dict]:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    table = pd.read_csv(features_path)
    required_columns = [schema["entity"], *schema["features"], schema["target"]]
    missing_columns = [column for column in required_columns if column not in table.columns]
    if missing_columns:
        raise ValueError(f"Feature table is missing columns: {', '.join(missing_columns)}")
    if table.empty:
        raise ValueError("Feature table contains no rows.")
    if table[required_columns].isna().any().any():
        raise ValueError("Feature table contains missing values.")
    return table, schema


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize the wine offline feature store.")
    parser.add_argument("--input", type=Path, default=Path("wine.csv"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/wine_features.csv")
    )
    parser.add_argument("--schema", type=Path, default=Path("feature_store/schema.json"))
    args = parser.parse_args()

    build_feature_table(args.input, args.output, args.schema)
    print(f"Feature table written to {args.output}")
    print(f"Feature schema written to {args.schema}")


if __name__ == "__main__":
    main()
