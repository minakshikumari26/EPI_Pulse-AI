import contextlib
import io
import unittest

import pandas as pd

from api.main import get_regions, get_simple_forecast
from src.alerting.alert_generator import generate_alert
from src.preprocessing.clean_data import load_and_clean_data
from src.preprocessing.feature_engineering import create_features
from src.risk_scoring.risk_score import calculate_risk


class CoreWorkflowTests(unittest.TestCase):
    def test_clean_data_returns_valid_dataframe(self):
        with contextlib.redirect_stdout(io.StringIO()):
            df = load_and_clean_data()

        self.assertFalse(df.empty)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df["date"]))
        self.assertEqual(df.duplicated().sum(), 0)
        self.assertEqual(df[["cases", "temperature", "humidity", "rainfall"]].isna().sum().sum(), 0)

    def test_feature_engineering_is_region_aware(self):
        with contextlib.redirect_stdout(io.StringIO()):
            df = create_features()

        first_rows = df.groupby("region").head(1)
        self.assertTrue(first_rows["previous_day_cases"].isna().all())
        self.assertTrue({"rolling_avg_cases", "previous_day_cases", "growth_rate"}.issubset(df.columns))

    def test_risk_scores_are_bounded_and_labeled(self):
        with contextlib.redirect_stdout(io.StringIO()):
            df = calculate_risk()

        self.assertTrue(df["risk_score"].between(0, 100).all())
        self.assertTrue(set(df["risk_level"]).issubset({"Low", "Medium", "High"}))

    def test_alert_generator_uses_risk_signals(self):
        self.assertEqual(generate_alert(risk_score=75), "High outbreak risk")
        self.assertEqual(generate_alert(risk_score=55), "Medium outbreak risk")
        self.assertEqual(generate_alert(cases=5, risk_score=10), "Low outbreak risk")

    def test_api_helpers_return_expected_shapes(self):
        regions = get_regions()["regions"]
        forecast = get_simple_forecast(region="Delhi")

        self.assertIn("Delhi", regions)
        self.assertEqual(len(forecast), 7)
        self.assertTrue(all(row["region"] == "Delhi" for row in forecast))


if __name__ == "__main__":
    unittest.main()
