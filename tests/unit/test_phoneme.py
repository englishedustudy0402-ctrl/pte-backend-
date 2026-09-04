import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from services.phoneme import (  # noqa: E402
    aligner_available,
    parse_timestamps,
    phoneme_pronunciation,
    word_pronunciation_score,
)


def make_voiced_signal(rate=16000, duration=0.5):
    """A clean voiced tone (simulates clear, stable articulation)."""
    t = np.arange(int(rate * duration)) / rate
    return (0.4 * np.sin(2 * np.pi * 180.0 * t)).astype(np.float32)


def make_silence(rate=16000, duration=0.5):
    return np.zeros(int(rate * duration), dtype=np.float32)


class PhonemeTest(unittest.TestCase):
    def test_parse_timestamps_dict(self):
        ts = [{"word": "the", "start": 0.0, "end": 0.2},
              {"word": "cat", "start": 0.2, "end": 0.4}]
        out = parse_timestamps(ts)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["word"], "the")

    def test_parse_timestamps_tuples(self):
        out = parse_timestamps([("hello", 0.0, 0.3)])
        self.assertEqual(out[0]["word"], "hello")

    def test_parse_timestamps_empty(self):
        self.assertEqual(parse_timestamps(None), [])
        self.assertEqual(parse_timestamps("bad"), [])

    def test_word_voiced_scores_above_zero(self):
        sig = make_voiced_signal()
        r = word_pronunciation_score(sig, 16000, {"word": "test", "start": 0.0, "end": 0.5})
        self.assertGreater(r["score01"], 0.3)

    def test_word_silence_scores_low(self):
        sig = make_silence()
        r = word_pronunciation_score(sig, 16000, {"word": "test", "start": 0.0, "end": 0.5})
        self.assertLess(r["score01"], 0.3)

    def test_phoneme_pronunciation_voiced(self):
        sig = make_voiced_signal(16000, 0.6)
        ts = [{"word": "the", "start": 0.0, "end": 0.2},
              {"word": "cat", "start": 0.2, "end": 0.4},
              {"word": "sat", "start": 0.4, "end": 0.6}]
        res = phoneme_pronunciation(sig, 16000, ts)
        self.assertIn(res["mode"], ("acoustic-fused", "forced-align"))
        self.assertEqual(res["words"], 3)
        self.assertGreater(res["pron90"], 10)
        self.assertLessEqual(res["pron90"], 90)

    def test_phoneme_pronunciation_empty(self):
        res = phoneme_pronunciation(None, 16000, [])
        self.assertEqual(res["words"], 0)
        self.assertEqual(res["pron90"], 0)

    def test_aligner_available_is_bool(self):
        self.assertIn(aligner_available(), (True, False))


if __name__ == "__main__":
    unittest.main(verbosity=1)
