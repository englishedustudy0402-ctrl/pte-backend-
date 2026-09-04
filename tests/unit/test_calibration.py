import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.calibration import (  # noqa: E402
    confidence,
    estimate,
    info_weight,
    online_update,
    recency_weight,
)


class CalibrationTest(unittest.TestCase):
    def test_info_weight_in_range(self):
        self.assertGreaterEqual(info_weight(0.0), 0.9)  # well-matched item is informative
        self.assertLessEqual(info_weight(0.0), 1.0)

    def test_recency_decay(self):
        self.assertAlmostEqual(recency_weight(0), 1.0)
        self.assertAlmostEqual(recency_weight(30, 30), 0.5, places=5)
        self.assertLess(recency_weight(120, 30), 0.1)

    def test_confidence_grows_with_n(self):
        self.assertEqual(confidence(0), 0.0)
        self.assertLess(confidence(1), confidence(20))
        self.assertLessEqual(confidence(1000), 1.0)

    def test_estimate_empty(self):
        r = estimate([])
        self.assertEqual(r["value"], 10)
        self.assertEqual(r["n"], 0)
        self.assertEqual(r["confidence"], 0.0)

    def test_estimate_single(self):
        r = estimate([{"score": 72, "difficulty_b": 0.0, "age_days": 0}])
        self.assertEqual(r["n"], 1)
        self.assertLessEqual(r["value"], 90)
        self.assertGreaterEqual(r["value"], 10)

    def test_estimate_recent_outweighs_stale(self):
        recent = estimate([{"score": 55, "difficulty_b": 0.0, "age_days": 0}])
        stale_mix = estimate([
            {"score": 55, "difficulty_b": 0.0, "age_days": 0},
            {"score": 88, "difficulty_b": 0.0, "age_days": 200},
        ])
        # the recent 55 should pull the mix down below the stale 88's influence
        self.assertLess(stale_mix["value"], 88)

    def test_online_update_folds_new_score(self):
        current = estimate([{"score": 60, "difficulty_b": 0.0, "age_days": 0}])
        updated = online_update(current, {"score": 90, "difficulty_b": 0.0, "age_days": 0})
        self.assertEqual(updated["n"], current["n"] + 1)
        self.assertGreater(updated["value"], current["value"])


if __name__ == "__main__":
    unittest.main(verbosity=1)
