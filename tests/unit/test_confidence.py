import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.confidence import (  # noqa: E402
    narrow,
    overall_band,
    points_band,
    sem,
    skill_band,
)


class ConfidenceTest(unittest.TestCase):
    def test_sem_decreases_with_n(self):
        self.assertGreater(sem(1), sem(50))

    def test_points_band_in_range(self):
        self.assertLessEqual(points_band(1.0), 15)
        self.assertGreaterEqual(points_band(0.1), 0.5)

    def test_skill_band_contains_value(self):
        b = skill_band(64, 12)
        self.assertLessEqual(b["min"], 64)
        self.assertGreaterEqual(b["max"], 64)
        self.assertLessEqual(b["width"], 15)
        self.assertGreaterEqual(b["confidence"], 0)
        self.assertLessEqual(b["confidence"], 1)
        self.assertEqual(b["n"], 12)

    def test_more_items_narrow_band(self):
        wide = skill_band(60, 2)["width"]
        tight = skill_band(60, 80)["width"]
        self.assertGreater(wide, tight)

    def test_irt_se_widens_band(self):
        # use n large enough that the ±band is not clamped to its 15 max
        self.assertGreater(
            skill_band(60, 60, se_theta=1.5)["width"],
            skill_band(60, 60, se_theta=0.0)["width"],
        )

    def test_overall_band_range(self):
        ob = overall_band(60, [{"width": 4}, {"width": 5}, {"width": 3}])
        self.assertLessEqual(ob["min"], 60)
        self.assertGreaterEqual(ob["max"], 60)

    def test_narrow_pulls_low_confidence_toward_mid(self):
        self.assertGreater(narrow(85, 0.1), 10)      # not floored
        self.assertGreater(narrow(85, 0.05), narrow(85, 0.01))  # less shrink more stays
        self.assertLess(narrow(85, 0.3), 85)         # shrunk below original


if __name__ == "__main__":
    unittest.main(verbosity=1)
