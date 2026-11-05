"""Unit tests for November engagement and confidence analysis modules."""
import math
import unittest

from engagement_segment_analysis import (
    assign_segment, build_report as build_engagement_report
)
from prediction_confidence_analysis import (
    analyze_record, build_report as build_confidence_report,
    extract_probabilities
)

class EngagementAnalysisTests(unittest.TestCase):
    def sample_students(self):
        return [
            {"total_clicks": 10, "active_days": 2, "avg_score": 40,
             "final_result": "Fail", "p_Fail": 0.7},
            {"total_clicks": 20, "active_days": 4, "avg_score": 60,
             "final_result": "Pass", "p_Fail": 0.2},
            {"total_clicks": 40, "active_days": 8, "avg_score": 80,
             "final_result": "Distinction", "p_Fail": 0.1},
        ]

    def test_segment_thresholds(self):
        self.assertEqual(assign_segment(0.5), "Low")
        self.assertEqual(assign_segment(1.0), "Medium")
        self.assertEqual(assign_segment(1.5), "High")

    def test_all_students_are_counted(self):
        report = build_engagement_report(self.sample_students())
        total = sum(item["student_count"] for item in report["segments"])
        self.assertEqual(total, 3)
        self.assertEqual(report["student_records_analyzed"], 3)

    def test_empty_engagement_data_is_rejected(self):
        with self.assertRaises(ValueError):
            build_engagement_report([])

class PredictionConfidenceTests(unittest.TestCase):
    def valid_record(self):
        return {
            "p_Distinction": 0.1, "p_Fail": 0.7,
            "p_Pass": 0.15, "p_Withdrawn": 0.05,
            "final_result": "Fail"
        }

    def test_probabilities_are_normalized(self):
        probabilities = extract_probabilities(self.valid_record())
        self.assertAlmostEqual(sum(probabilities), 1.0)

    def test_prediction_and_correctness(self):
        result = analyze_record(self.valid_record())
        self.assertEqual(result["predicted_class"], "Fail")
        self.assertTrue(result["correct"])
        self.assertGreaterEqual(result["confidence"], 0.0)

    def test_invalid_probability_is_rejected(self):
        record = self.valid_record()
        record["p_Fail"] = float("nan")
        with self.assertRaises(ValueError):
            extract_probabilities(record)

    def test_confidence_report_counts_valid_rows(self):
        report = build_confidence_report([self.valid_record(), {"bad": "row"}])
        self.assertEqual(report["records_analyzed"], 1)
        self.assertEqual(report["records_skipped"], 1)
        self.assertEqual(report["overall_observed_accuracy"], 1.0)

if __name__ == "__main__":
    unittest.main()
