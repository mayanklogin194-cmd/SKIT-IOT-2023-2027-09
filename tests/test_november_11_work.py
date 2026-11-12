"""Tests for November 11 intervention and class-error analysis modules."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from intervention_recommendations import (
    risk_band, engagement_band, recommend_action, build_recommendations
)
from class_error_analysis import analyze_matrix, build_report

class InterventionRecommendationTests(unittest.TestCase):
    def test_risk_boundaries(self):
        self.assertEqual(risk_band(0.70), "High")
        self.assertEqual(risk_band(0.40), "Medium")
        self.assertEqual(risk_band(0.10), "Low")

    def test_engagement_uses_clicks_and_active_days(self):
        self.assertEqual(engagement_band(20, 3), "Low")
        self.assertEqual(engagement_band(200, 12), "Medium")
        self.assertEqual(engagement_band(500, 30), "High")

    def test_high_risk_low_engagement_action(self):
        action = recommend_action("High", "Low", 35)
        self.assertIn("Contact student", action)

    def test_report_counts_every_record(self):
        rows = [
            {"id_student": 1, "risk": 0.8, "total_clicks": 20, "active_days": 2},
            {"id_student": 2, "risk": 0.2, "total_clicks": 500, "active_days": 30},
        ]
        report = build_recommendations(rows)
        self.assertEqual(report["records_processed"], 2)
        self.assertEqual(report["risk_counts"]["High"], 1)

class ClassErrorAnalysisTests(unittest.TestCase):
    def test_per_class_counts_and_lowest_recall(self):
        result = analyze_matrix(["Pass", "Fail"], [[8, 2], [4, 6]])
        self.assertEqual(result["sample_count"], 20)
        self.assertEqual(result["lowest_recall_class"], "Fail")
        self.assertEqual(result["class_results"][0]["correct"], 8)

    def test_rejects_non_square_matrix(self):
        with self.assertRaises(ValueError):
            analyze_matrix(["Pass", "Fail"], [[1, 2], [3]])

    def test_build_report_skips_models_without_matrix(self):
        payload = {
            "selected": "A",
            "classes": ["Pass", "Fail"],
            "models": {
                "A": {"confusion_matrix": [[8, 2], [4, 6]]},
                "B": {}
            }
        }
        report = build_report(payload)
        self.assertIn("A", report["models"])
        self.assertNotIn("B", report["models"])

if __name__ == "__main__":
    unittest.main()
