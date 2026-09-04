import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.adaptive import (  # noqa: E402
    band_label,
    next_label,
    select_next,
    select_next_online,
    theta_from_history,
)


class AdaptiveTest(unittest.TestCase):
    def test_band_label_splits(self):
        self.assertEqual(band_label(-2), "easy")
        self.assertEqual(band_label(0), "medium")
        self.assertEqual(band_label(2), "hard")

    def test_next_label_no_history(self):
        self.assertEqual(next_label(-2), "easy")
        self.assertEqual(next_label(0), "medium")
        self.assertEqual(next_label(2), "hard")

    def test_hysteresis_holds_easy(self):
        # on easy a weak-but-not-awful θ (-0.5) stays easy (below -0.75+0.35)
        self.assertEqual(next_label(-0.5, "easy"), "easy")
        # a genuinely medium-range ability is served medium, not held back
        self.assertEqual(next_label(0.2, "easy"), "medium")

    def test_hysteresis_holds_medium(self):
        # borderline ability on medium stays medium (no bounce)
        self.assertEqual(next_label(0.2, "medium"), "medium")
        self.assertEqual(next_label(-0.2, "medium"), "medium")
        # clearly past the edge -> break out
        self.assertEqual(next_label(1.2, "medium"), "hard")
        self.assertEqual(next_label(-1.2, "medium"), "easy")

    def test_hysteresis_holds_hard(self):
        # still clearly strong on hard stays hard; only drop when clearly under
        self.assertEqual(next_label(0.5, "hard"), "hard")
        self.assertEqual(next_label(0.2, "hard"), "medium")

    def test_strong_history_gives_hard(self):
        hist = [{"difficulty": "medium", "outcome": 1}] * 8
        r = select_next(hist, previous_label="medium")
        self.assertGreater(r["theta"], 0)
        self.assertIn(r["difficulty"], ("medium", "hard"))

    def test_weak_history_gives_easy(self):
        hist = [{"difficulty": "medium", "outcome": 0}] * 8
        r = select_next(hist, previous_label="medium")
        self.assertLess(r["theta"], 0)
        self.assertIn(r["difficulty"], ("easy", "medium"))

    def test_online_update_responds_to_outcome(self):
        up = select_next_online(0.0, 1.0, "hard", previous_label="medium")
        down = select_next_online(0.0, 0.0, "hard", previous_label="medium")
        self.assertGreater(up["theta"], down["theta"])

    def test_theta_from_history_shape(self):
        est = theta_from_history([{"difficulty": "easy", "outcome": 1}])
        self.assertIn("theta", est)
        self.assertEqual(est["n"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=1)
