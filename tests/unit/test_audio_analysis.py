import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.audio_analysis import (  # noqa: E402
    analyze,
    fluency_from_features,
    pronunciation_from_features,
    _autocorr_f0,
)


def empty_audio():
    return {
        "available": False, "duration_s": 0.0, "voiced_s": 0.0,
        "voicing_ratio": 0.0, "speech_rate_sps": 0.0, "speech_rate_wpm": 0.0,
        "pauses": [], "long_pauses": 0, "mean_rms": 0.0,
        "f0_mean_hz": None, "f0_std_hz": 0.0, "f0_semitone_sd": 0.0,
        "clipping_ratio": 0.0, "confidence": 0.0, "words": 0,
    }


class AudioAnalysisUnitTest(unittest.TestCase):
    def test_empty_audio_degrades(self):
        r = analyze(b"", "audio/webm", transcript="")
        self.assertIs(r["available"], False)
        self.assertEqual(r["speech_rate_sps"], 0.0)
        f = fluency_from_features(r)
        self.assertEqual(f["fluency01"], 0.0)

    def test_fluency_from_features_no_audio(self):
        f = fluency_from_features(empty_audio())
        self.assertEqual(f["fluency01"], 0.0)

    def test_pron_from_features_no_audio(self):
        p = pronunciation_from_features(empty_audio())
        self.assertEqual(p["pron01"], 0.0)

    def test_long_pauses_penalise_fluency(self):
        audio = empty_audio()
        audio["available"] = True
        audio["pauses"] = [4.0, 3.5]          # two long (legal) pauses
        audio["speech_rate_sps"] = 2.5
        audio["voicing_ratio"] = 0.8
        audio["long_pauses"] = 2
        f = fluency_from_features(audio)
        self.assertLess(f["fluency01"], 1.0)
        self.assertEqual(f["long_pauses"], 2)

    def test_slow_rate_penalises_fluency(self):
        audio = empty_audio()
        audio["available"] = True
        audio["pauses"] = []
        audio["speech_rate_sps"] = 1.0         # very slow
        audio["voicing_ratio"] = 0.8
        f = fluency_from_features(audio)
        self.assertLess(f["fluency01"], 1.0)

    def test_flat_intonation_penalises_pronunciation(self):
        audio = empty_audio()
        audio["available"] = True
        audio["f0_semitone_sd"] = 0.2          # flat intonation
        audio["clipping_ratio"] = 0.0
        p = pronunciation_from_features(audio)
        self.assertLess(p["pron01"], 1.0)

    def test_autocorr_f0_returns_frequency(self):
        # 200 Hz sinusoid sampled at 16 kHz
        import numpy as np
        rate = 16000
        t = np.arange(int(rate * 0.1)) / rate
        frame = (0.5 * np.sin(2 * np.pi * 200.0 * t)).astype(np.float32)
        f0 = _autocorr_f0(frame, rate)
        self.assertIsNotNone(f0)
        self.assertAlmostEqual(f0, 200.0, delta=5.0)


if __name__ == "__main__":
    unittest.main(verbosity=1)
