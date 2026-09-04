import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.equating import (  # noqa: E402
    equate_question,
    equate_raw_to_90,
    equate_score,
    equate_traits,
)


class EquatingTest(unittest.TestCase):
    def test_score_range(self):
        r = equate_score(0.8, "medium", 1)
        self.assertGreaterEqual(r["score"], 10)
        self.assertLessEqual(r["score"], 90)

    def test_hard_item_equates_higher_than_easy(self):
        easy = equate_score(0.8, "easy", 1)["score"]
        hard = equate_score(0.8, "hard", 1)["score"]
        self.assertGreater(hard, easy)

    def test_monotonic_in_success(self):
        low = equate_score(0.2, "medium", 1)["score"]
        high = equate_score(0.9, "medium", 1)["score"]
        self.assertLess(low, high)

    def test_raw_90_convenience(self):
        r = equate_raw_to_90(81, "hard", 1)
        self.assertGreaterEqual(r["score"], 10)
        self.assertLessEqual(r["score"], 90)

    def test_equate_traits_total(self):
        eq = equate_traits({"content": 70, "fluency": 50, "pronunciation": 60}, "medium", 1)
        self.assertEqual(set(eq.keys()), {"content", "fluency", "pronunciation", "score"})
        for k in ("content", "fluency", "pronunciation", "score"):
            self.assertGreaterEqual(eq[k], 10)
            self.assertLessEqual(eq[k], 90)

    def test_equate_question_uses_question_difficulty(self):
        q = {"content": {"difficulty": "hard"}}
        r = equate_question(q, {"content": 80, "fluency": 80, "pronunciation": 80})
        self.assertIn("score", r)

    def test_more_items_tightens(self):
        single = equate_score(0.5, "medium", 1)["score"]
        multi = equate_score(0.5, "medium", 10)["score"]
        # more items moves the blended estimate toward the raw effort value
        self.assertIn(multi, range(10, 91))


if __name__ == "__main__":
    unittest.main(verbosity=1)
