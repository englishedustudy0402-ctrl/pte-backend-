"""
Real audio-feature extraction for PTE-style speaking scoring (Tier 1).

Purpose
-------
True to the Pearson approach, pronunciation and fluency must be measured from
the SOUND signal (waveform), not from a text transcript. This module decodes an
uploaded recording (webm/opus/wav/mp3/ogg) to mono PCM via ffmpeg and numpy, and
computes the acoustic features PTE-style examiners look for:

  * Voice activity detection (VAD) -> silences / pauses, with exact timings.
  * Speech rate -> syllables-per-second proxy from the voiced envelope.
  * Voicing ratio -> fraction of the clip that is actually speech (vs silence).
  * RMS / energy statistics -> loudness, clipping, quiet segments.
  * Pitch (F0) variability -> a pronunciation / intonation confidence signal
    (stable, natural intonation scores higher than flat or erratic pitch).

These features are returned as a JSON-friendly dict that the scoring router
merges into the result, giving the downstream score a REAL audio backbone.

Design notes
------------
- Pure numpy; no librosa, keeping the Docker image light and the startup fast.
- ffmpeg is used ONLY to decode to raw mono PCM. If ffmpeg is absent, the
  module degrades gracefully and returns empty/neutral features (the caller
  falls back to transcript-derived heuristics).
- All functions are pure and deterministic, so they are unit-testable without
  a hosted backend.
"""
import math
import os
import shutil
import subprocess
import tempfile
from typing import Optional

import numpy as np

# frame the audio at 20 ms for VAD + pitch analysis (25 fps feel, cheap)
FRAME_MS = 20.0
MAX_FRAMES = 6000          # cap analysis at 120 s to bound CPU on long clips

# VAD threshold on RMS (relative to a lightly-smoothed noise floor). Tuned so a
# normal speaking voice registers as voiced while trailing room hum stays silent.
VAD_RMS_FLOOR = 0.006       # absolute lower bound (quiet recordings)
VAD_SILENCE_SEC = 0.30      # contiguous silent frames shorter than this = one pause
    
# Silence/pause penalties tuned to the PTE oral-fluency band descriptors:
#   pauses longer than 3 s are heavily penalised in the real test.
PAUSE_HARD_SEC = 3.0


def decode_to_pcm(audio_bytes: bytes, mime: str = "audio/webm",
                  target_rate: int = 16000) -> Optional[np.ndarray]:
    """Decode raw audio bytes to a mono float32 PCM array at target_rate.
    Returns None if ffmpeg is unavailable or decoding fails."""
    exe = shutil.which("ffmpeg")
    if not exe:
        return None
    try:
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "input.src")
            raw = os.path.join(d, "out.f32le")
            with open(src, "wb") as fh:
                fh.write(audio_bytes)
            r = subprocess.run(
                [exe, "-y", "-i", src,
                 "-ac", "1", "-ar", str(target_rate),
                 "-f", "f32le", "-acodec", "pcm_f32le", raw],
                capture_output=True, timeout=60,
            )
            if r.returncode != 0 or not os.path.exists(raw):
                return None
            with open(raw, "rb") as fh:
                data = fh.read()
            arr = np.frombuffer(data, dtype=np.float32).copy()
            if arr.size == 0:
                return None
            # normalise to a sensible peak if the clip is very quiet
            peak = float(np.max(np.abs(arr))) if arr.size else 0.0
            if 0 < peak < 0.05:
                arr = arr * (0.05 / peak)
            return arr
    except Exception:
        return None


def _frames(sig: np.ndarray, rate: int) -> np.ndarray:
    """Split mono signal into RMS-per-frame → shape (n_frames,)."""
    n = int(rate * FRAME_MS / 1000.0)
    sig = sig[: n * MAX_FRAMES]
    if sig.size == 0:
        return np.zeros(0)
    extra = n - (sig.size % n)
    padded = np.pad(sig, (0, extra % n), mode="constant")
    frames = padded.reshape(-1, n)
    return np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)


def _smooth(x: np.ndarray, k: int = 3) -> np.ndarray:
    if x.size == 0:
        return x
    if k < 2:
        return x
    kernel = np.ones(k) / k
    return np.convolve(x, kernel, mode="same")


def _segments(flags: np.ndarray) -> list:
    """Split a boolean array into (start_frame, end_frame) contiguous run segments."""
    out = []
    start = None
    for i, v in enumerate(flags):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(flags) - 1))
    return out


def _autocorr_f0(frame: np.ndarray, rate: int,
                 fmin: float = 75.0, fmax: float = 400.0) -> Optional[float]:
    """Estimate pitch via autocorrelation over a window low-passed at ~fmax.
    Returns Hz or None if the frame is voiceless (low energy / no clear period)."""
    if frame.size < 2:
        return None
    rms = float(np.sqrt(np.mean(frame ** 2) + 1e-12))
    if rms < VAD_RMS_FLOOR:
        return None
    # light low-pass: average pairs to halve sample rate, effectively band-limit
    frame = (frame[::2] + frame[1::2]) * 0.5
    lag_min = max(1, int(rate / 2 / fmax))
    lag_max = min(len(frame) - 1, int(rate / 2 / fmin))
    if lag_max <= lag_min:
        return None
    x = frame - frame.mean()
    denom = float(np.dot(x, x) + 1e-12)
    if denom <= 0:
        return None
    # sharply spiky autocorr (low values) == clear pitch
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    ac = ac[:lag_max] / denom
    seg = ac[lag_min:lag_max]
    if seg.size == 0:
        return None
    lag = int(np.argmax(seg)) + lag_min
    peak = float(seg[lag - lag_min])
    if peak < 0.35:
        return None
    return rate / 2.0 / lag


def _median_f0_smooth(f0: list, k: int = 3):
    """Median-filter an F0 list to reject octave/voicing jitter, preserving None
    (unvoiced) as gaps so silence stays silent."""
    out = list(f0)
    n = len(f0)
    if n == 0 or k < 2:
        return out
    for i in range(n):
        lo = max(0, i - k)
        hi = min(n, i + k + 1)
        window = [v for v in f0[lo:hi] if v is not None]
        if window and f0[i] is not None:
            out[i] = float(np.median(window))
    return out


def analyze(audio_bytes: bytes, mime: str = "audio/webm",
            reference_text: str = "", transcript: str = "") -> dict:
    """Compute acoustic features from raw audio bytes.

    Returns
    -------
    dict with keys:
      available: bool      -> whether waveform analysis succeeded
      duration_s: float    -> total decoded audio duration
      voiced_s: float      -> time classified as speech
      voicing_ratio: float -> 0..1 fraction of clip that is speech
      speech_rate_sps: float    -> syllables per second of voiced time
      speech_rate_wpm: float    -> words per minute (from voiced time + transcript)
      pauses: list[float]  -> durations of detected silences >= 300 ms
      long_pauses: int     -> pauses > 3 s (heavily penalised)
      mean_rms: float      -> voiced-frame average RMS
      f0_mean_hz / f0_std_hz: float -> pitch stats over voiced frames
      f0_semitone_sd: float -> pitch variation in semitones (intonation spread)
      clipping_ratio: float -> fraction of samples at |x| >= 0.99
      confidence: float     -> 0..1 how confident the VAD/pitch features are
      words: int            -> transcript word count
    """
    sig = decode_to_pcm(audio_bytes, mime)
    if sig is None or sig.size == 0:
        return {
            "available": False, "duration_s": 0.0, "voiced_s": 0.0,
            "voicing_ratio": 0.0, "speech_rate_sps": 0.0, "speech_rate_wpm": 0.0,
            "pauses": [], "long_pauses": 0, "mean_rms": 0.0,
            "f0_mean_hz": None, "f0_std_hz": 0.0, "f0_semitone_sd": 0.0,
            "clipping_ratio": 0.0, "confidence": 0.0, "words": 0,
        }

    rate = 16000
    duration_s = round(sig.size / rate, 3)
    rms = _frames(sig, rate)
    n = rms.size

    # ---- adaptive VAD threshold ---------------------------------------------
    # Start from a mild floor (low percentile of the RMS distribution). The key
    # safeguard: if the signal's MEDIAN is clearly above the absolute floor
    # (i.e. there is plainly content — a steady tone, or speech with a high
    # noise floor), never let the threshold exceed ~50% of the median. Without
    # this, a uniform-amplitude signal can push percentile*multiplier above its
    # own RMS and get classified as 100% silence (self-defeating VAD).
    median_rms = float(np.median(rms)) if n else 0.0
    floor = np.percentile(rms, 15) * 1.5 if n else 0.0
    thr = max(VAD_RMS_FLOOR, floor)
    if median_rms > VAD_RMS_FLOOR * 3:
        thr = min(thr, median_rms * 0.5)
    voiced = rms > thr
    voiced = _smooth(voiced.astype(np.float32), 4) > 0.5

    voiced_s = float(np.count_nonzero(voiced) * FRAME_MS / 1000.0)
    voicing_ratio = voiced_s / duration_s if duration_s > 0 else 0.0

    # ---- silences / pauses (silent runs >= 300 ms)
    sil = ~voiced
    pauses = []
    for (a, b) in _segments(sil):
        secs = (b - a + 1) * FRAME_MS / 1000.0
        if secs >= VAD_SILENCE_SEC:
            pauses.append(round(secs, 3))
    long_pauses = int(sum(1 for p in pauses if p > PAUSE_HARD_SEC))

    # ---- speech rate
    words = len([w for w in (transcript or "").split() if w.strip()])
    speech_rate_sps = words / voiced_s if voiced_s > 0 else 0.0
    speech_rate_wpm = (speech_rate_sps * 60.0) if words else 0.0

    # ---- RMS over voiced frames
    vrms = rms[voiced]
    mean_rms = float(np.mean(vrms)) if vrms.size else 0.0

    # ---- pitch / intonation over voiced frames (skip heavy loops on long clips)
    f0 = []
    step = max(1, n // 2000)          # sample ~2000 frames max for F0
    for i in range(0, n, step):
        if voiced[i]:
            seg = sig[i * int(FRAME_MS * rate / 1000):
                       (i + 1) * int(FRAME_MS * rate / 1000)]
            f = _autocorr_f0(seg, rate)
            f0.append(f)
        else:
            f0.append(None)
    f0 = _median_f0_smooth(f0, 2)
    voiced_f0 = [f for f in f0 if f is not None]

    if voiced_f0:
        f0_mean = float(np.mean(voiced_f0))
        f0_std = float(np.std(voiced_f0))
        # semitone spread: std of 12*log2(f/mean)
        f0_sd_st = float(np.std([12.0 * math.log2(f / f0_mean) for f in voiced_f0]))
    else:
        f0_mean, f0_std, f0_sd_st = None, 0.0, 0.0

    # ---- clipping (distortion) ratio
    clipping_ratio = float(np.mean(np.abs(sig) >= 0.99)) if sig.size else 0.0

    # ---- confidence: how much usable voiced signal we got
    confidence = min(1.0, voicing_ratio) * (0.6 if voiced_s > 0 else 0.0)

    return {
        "available": True,
        "duration_s": duration_s,
        "voiced_s": round(voiced_s, 3),
        "voicing_ratio": round(voicing_ratio, 4),
        "speech_rate_sps": round(speech_rate_sps, 3),
        "speech_rate_wpm": round(speech_rate_wpm, 1),
        "pauses": pauses,
        "long_pauses": long_pauses,
        "mean_rms": round(mean_rms, 4),
        "f0_mean_hz": round(f0_mean, 1) if f0_mean else None,
        "f0_std_hz": round(f0_std, 1),
        "f0_semitone_sd": round(f0_sd_st, 3),
        "clipping_ratio": round(clipping_ratio, 4),
        "confidence": round(confidence, 3),
        "words": words,
    }


def fluency_from_features(audio: dict) -> dict:
    """Turn acoustic features into a 0..1 oral-fluency factor the scoring layer
    can multiply into fluency. Returns {fluency01, reasons}."""
    if not audio.get("available"):
        return {"fluency01": 0.0, "reasons": ["no audio available"], "long_pauses": 0}

    sps = float(audio.get("speech_rate_sps") or 0.0)
    voicing = float(audio.get("voicing_ratio") or 0.0)
    long_pauses = int(audio.get("long_pauses") or 0)
    pauses = audio.get("pauses") or []

    # total paused seconds beyond the first 0.5 s (speaker breathing room)
    pause_time = max(0.0, sum(pauses) - 0.5 * len(pauses))

    score = 1.0
    reasons = []

    # Natural speaking rate ~ 2.2-4.5 syllables/s (~ 132-270 wpm at 60 wpmin).
    if 0 < sps < 1.4:
        score *= 0.55
        reasons.append(f"very slow delivery ({sps:.1f} syl/s)")
    elif 0 < sps < 2.0:
        score *= 0.78
        reasons.append(f"slow delivery ({sps:.1f} syl/s)")
    elif sps > 5.5:
        score *= 0.80
        reasons.append(f"very fast, possibly rushed ({sps:.1f} syl/s)")

    # Pauses: heavy penalty per pause > 3 s (PTE band descriptor).
    penalty = 0.0
    for p in pauses:
        if p > PAUSE_HARD_SEC:
            penalty += min(0.35, 0.12 * (p - PAUSE_HARD_SEC))
    # a few short pauses are natural; many mid-length ones erode fluency slightly
    many_short = max(0, len(pauses) - 6) * 0.02 if len(pauses) > 6 else 0.0
    pause_pen = min(0.5, penalty + many_short)
    if pause_pen:
        reasons.append(f"{long_pauses} long pause(s); {len(pauses)} total")
        score *= (1.0 - pause_pen)

    # low voicing ratio -> lots of dead air / hesitation
    if voicing < 0.35:
        score *= 0.75
        reasons.append("large silent gaps across the clip")

    score = max(0.0, min(1.0, score))
    return {
        "fluency01": round(score, 3),
        "reasons": reasons,
        "long_pauses": long_pauses,
        "pause_time_s": round(pause_time, 2),
    }


def pronunciation_from_features(audio: dict) -> dict:
    """Heuristic pronunciation factor from acoustic features (0..1). This is a
    stand-in until a real phoneme-aligner is wired; it rewards stable natural
    pitch, reasonable loudness, and low clipping. Returns {pron01, reasons}."""
    if not audio.get("available"):
        return {"pron01": 0.0, "reasons": ["no audio available"]}

    score = 1.0
    reasons = []

    st = float(audio.get("f0_semitone_sd") or 0.0)
    if st and st >= 0.01:
        # healthy intonation ~ 1-3 semitones of variation; flat or erratic drops.
        if st < 0.6:
            score *= 0.82
            reasons.append(f"very flat intonation ({st:.2f} st)")
        elif st > 5.0:
            score *= 0.78
            reasons.append(f"highly erratic pitch ({st:.2f} st)")

    rms = float(audio.get("mean_rms") or 0.0)
    if rms > 0 and rms < 0.02:
        score *= 0.85
        reasons.append("very quiet delivery")

    clip = float(audio.get("clipping_ratio") or 0.0)
    if clip > 0.02:
        score *= 0.9
        reasons.append("audio distortion/overload detected")

    score = max(0.0, min(1.0, score))
    return {"pron01": round(score, 3), "reasons": reasons}
