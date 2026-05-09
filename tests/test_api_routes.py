import unittest

from api.main import (
    get_cases,
    get_regions,
    get_risk,
    get_simple_forecast,
    get_spikes,
    health_check,
    metrics,
)
from api.routes.prediction_routes import predict_outbreak
from api.schemas.prediction_schema import DiseaseInput


class ApiRouteTests(unittest.TestCase):
    def test_health(self):
        response = health_check()

        self.assertIn(response["status"], {"healthy", "degraded"})
        self.assertTrue(response["data_available"])

    def test_regions(self):
        response = get_regions()

        self.assertIn("Delhi", response["regions"])

    def test_cases_filter(self):
        records = get_cases(region="Delhi", start_date=None, end_date=None)

        self.assertGreater(len(records), 0)
        self.assertTrue(all(record["region"] == "Delhi" for record in records))

    def test_risk(self):
        records = get_risk()

        self.assertGreater(len(records), 0)
        self.assertIn("risk_score", records[0])

    def test_spikes(self):
        records = get_spikes()

        self.assertGreater(len(records), 0)
        self.assertIn("z_score", records[0])

    def test_simple_forecast(self):
        records = get_simple_forecast(region="Delhi")

        self.assertEqual(len(records), 7)

    def test_predict(self):
        result = predict_outbreak(
            DiseaseInput(
                region="Delhi",
                cases=60,
                temperature=32,
                humidity=88,
                rainfall=12,
            )
        )

        self.assertEqual(result["region"], "Delhi")
        self.assertIn("risk_score", result)
        self.assertIn("alert", result)

    def test_metrics(self):
        response = metrics()

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"epipulse", response.body)


if __name__ == "__main__":
    unittest.main()
