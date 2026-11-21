import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.feature_store import build_feature_table, load_feature_table


class FeatureStoreTests(unittest.TestCase):
    def test_builds_table_with_stable_entity_and_schema(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.csv"
            output = root / "feature_store" / "features.csv"
            schema_path = root / "feature_store" / "schema.json"
            pd.DataFrame({"alcohol": [12.0, 13.0], "target": [0, 1]}).to_csv(
                source, index=False
            )

            build_feature_table(source, output, schema_path)
            table, schema = load_feature_table(output, schema_path)

            self.assertEqual(table["wine_id"].tolist(), [0, 1])
            self.assertEqual(schema["features"], ["alcohol"])
            self.assertEqual(schema["row_count"], 2)

    def test_rejects_missing_target(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.csv"
            pd.DataFrame({"alcohol": [12.0]}).to_csv(source, index=False)

            with self.assertRaisesRegex(ValueError, "target"):
                build_feature_table(
                    source, root / "features.csv", root / "schema.json"
                )


if __name__ == "__main__":
    unittest.main()
