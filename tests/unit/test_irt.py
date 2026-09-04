import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.irt import (  # noqa: E402
    ability_from_observations,
    difficulty_logit,
    difficulty_weight,
    observations_to_ruler,
    p_correct,
    question_difficulty_logit,
    ruler_map_degree,
)


class IrtTest(unittest.TestCase):
    def test_icc_probability_range(self):
        # P is 0..1 and increases with ability
        self.assertTrue(p_correct(0, 0) <= 0.5001)
        self.assertTrue(p_correct(2, 0) > p_correct(-2, 0))
        self.assertAlmostEqual(p_correct(0, 0), 0.5, places=4)

    def test_difficulty_anchor_order(self):
        easy = difficulty_logit("easy")
        med = difficulty_logit("medium")
        hard = difficulty_logit("hard")
        self.assertLess(easy, med)
        self.assertLess(med, hard)

    def test_question_difficulty_from_content(self):
        q = {"content": {"difficulty": "hard"}}
        self.assertEqual(question_difficulty_logit(q), difficulty_logit("hard"))

    def test_ability_recovers_known_true_theta(self):
        # simulate a candidate with true θ = 1.2 answering 30 varied items
        import math
        true_theta = 1.2
        obs = []
        import random
        random.seed(7)
        for _ in range(30):
            b = random.uniform(-2, 2)
            p = 1.0 / (1.0 + math.exp(-(true_theta - b)))
            outcome = 1.0 if random.random() < p else 0.0
            obs.append({"b": b, "outcome": outcome})
        est = ability_from_observations(obs)
        self.assertIsNotNone(est["theta"])
        self.assertAlmostEqual(est["theta"], true_theta, delta=1.0)

    def test_all_correct_stays_finite(self):
        obs = [{"b": 0.0, "outcome": 1.0}] * 10
        est = ability_from_observations(obs)
        self.assertLess(est["theta"], 8.0)  # not +inf, clamped finite

    def test_ruler_monotonic(self):
        lo = ruler_map_degree(-4)
        mid = ruler_map_degree(0)
        hi = ruler_map_degree(4)
        self.assertLess(lo, mid)
        self.assertLess(mid, hi)
        self.assertEqual(mid, 50)
        self.assertGreaterEqual(lo, 10)
        self.assertLessEqual(hi, 90)
        # full 10..90 range is achievable across the logistic curve
        self.assertLess(lo, 30)
        self.assertGreater(hi, 70)

    def test_difficulty_weight_rewards_hard(self):
        easy = difficulty_weight(0.9, "easy")
        hard = difficulty_weight(0.9, "hard")
        self.assertGreater(hard, easy)
        self.assertGreaterEqual(hard, 0.9)

    def test_observations_to_ruler(self):
        obs = [{"b": 0.0, "outcome": 1.0}] * 5 + [{"b": 1.5, "outcome": 1.0}] * 5
        est, score = observations_to_ruler(obs)
        self.assertGreater(score, 50)  # strong performer on medium+hard items


if __name__ == "__main__":
    unittest.main(verbosity=1)
