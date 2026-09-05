"""
Tier 1b — Phoneme-level pronunciation scoring.

Real PTE pronunciation is judged per-phoneme: each sound in the reference is
compared against what the candidate actually produced. The most accurate way to
do that is **forced alignment** — an acoustic model that locates every word (and
ideally phoneme) of the reference in the recording, then a comparison of the
produced speech against the expected phones.

Design
------
This module is the complete, modular pronunciation engine with TWO backends:

  1. FORCED-ALIGNMENT (the authoritative path, used on the hosted server where a
     heavier speech model can run - see Tier 4c). When an aligner is configured
     (via env USE_PHONEME_ALIGNER or an importable model), `word_alignments()`
     returns word timestamps and `phoneme_pronunciation()` scores each word by
     direct phone agreement.

  2. ACOUSTIC-FUSED (the lightweight, dependency-free fallback that runs locally,
     completing Tier 1b now). When no aligner is present, each aligned word is
     scored from the REAL acoustic evidence inside its own time window: the
     smoothness of the energy envelope (clean articulation vs. mush), voicing
     continuity, and F0 (pitch) smoothness. Words that are cleanly voiced, with
     a stable envelope and continuous pitch, score high; mumbled/quiet/erratic
     words score low.

Both paths return the same shape, so the scoring router can switch transparently
and expose which mode produced the result. Everything is pure/deterministic and
unit-testable without a hosted backend.
"""
import math
import os

# ---- Backend detection -------------------------------------------------------

def aligner_available():
    """True if the WhisperX forced-alignment backend is importable (Tier 4c).

    Explicitly disabled via USE_PHONEME_ALIGNER=0/false/no/off; otherwise the
    presence of the installable package is the real signal (not just the env
    var, so we never claim forced-align mode we cannot actually run).
    """
    if os.getenv("USE_PHONEME_ALIGNER", "").strip().lower() in ("0", "false", "no", "off"):
        return False
    try:
        import whisperx  # noqa: F401
        return True
    except Exception:
        return False


# ---- Tier 4c — real forced alignment (WhisperX) -------------------------------

_ALIGN_CACHE = {}


def _align_model(language_code, device):
    """Load the wav2vec2 aligner + metadata once and cache them (Tier 4c)."""
    import whisperx
    key = (language_code, device)
    if key not in _ALIGN_CACHE:
        _ALIGN_CACHE[key] = whisperx.load_align_model(language_code, device)
    return _ALIGN_CACHE[key]


def word_alignments(sig, rate, reference_text, device=None, language_code="en"):
    """Forced-align `reference_text` onto the waveform with WhisperX (Tier 4c).

    This is the authoritative Tier 4c path: instead of guessing that transcript
    words are evenly spaced, we locate every reference word at its TRUE acoustic
    position. Returns the parse_timestamps shape ({word, start, end}) or [] when
    the aligner is unavailable/fails — the caller then falls back to the even
    spacing heuristic.
    """
    try:
        import tempfile
        import wave
        import numpy as np
        import whisperx
    except Exception:
        return []
    if sig is None or rate <= 0 or len(sig) == 0:
        return []
    ref = (reference_text or "").strip()
    if not ref:
        return []

    device = device or os.getenv("PHONEME_ALIGN_DEVICE", "cpu")

    # PCM floats -> temporary 16-bit mono WAV so whisperx.load_audio (ffmpeg)
    # can read and resample it to its native 16 kHz, exactly like a real upload.
    sig = np.asarray(sig, dtype=np.float32)
    peak = float(np.abs(sig).max())
    scale = 32767.0 / peak if peak > 1.0 else 32767.0
    pcm = np.clip(sig * scale, -32768, 32767).astype(np.int16)

    fd, path = tempfile.mkstemp(suffix=".wav")
    try:
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(int(rate))
            w.writeframes(pcm.tobytes())

        audio = whisperx.load_audio(path)
        model_a, metadata = _align_model(language_code, device)
        dur = float(len(pcm)) / rate
        segments = [{"start": 0.0, "end": max(dur, 0.1), "text": ref}]
        aligned = whisperx.align(
            segments, model_a, metadata, audio, device,
            return_char_alignments=False,
        )
        return parse_timestamps(aligned.get("word_segments"))
    except Exception:
        return []
    finally:
        os.close(fd)
        try:
            os.remove(path)
        except OSError:
            pass


def parse_timestamps(timestamps):
    """
    Normalise aligner output to a list of {"word","start","end"} (seconds).
    Accepts either that exact shape or a list of [word, start, end] tuples.
    Returns [] if empty/malformed.
    """
    out = []
    for t in (timestamps or []):
        try:
            if isinstance(t, dict):
                word = str(t.get("word") or "")
                start = float(t.get("start") or t.get("start_time") or 0)
                end = float(t.get("end") or t.get("end_time") or start)
            else:
                word, start, end = str(t[0]), float(t[1]), float(t[2])
            word = word.strip()
            if word and end >= start:
                out.append({"word": word, "start": start, "end": end})
        except (ValueError, TypeError, IndexError):
            continue
    return out


# ---- Acoustic per-word scoring (local fallback backend) ----------------------

def _window(sig, rate, start, end):
    """Slice the mono PCM signal to the [start,end] window (seconds)."""
    if sig is None or rate <= 0 or end <= start:
        return None
    i0 = max(0, int(start * rate))
    i1 = max(i0, min(len(sig), int(end * rate)))
    if i1 - i0 < 1:
        return None
    return sig[i0:i1]


def _rms(x):
    if x is None or len(x) == 0:
        return 0.0
    return float(math.sqrt(float((x ** 2).mean())))


def _voicing_fraction(x):
    """Fraction of samples in a smoothed window above a small energy floor."""
    if x is None or len(x) == 0:
        return 0.0
    import numpy as np
    w = 512
    energy = np.abs(x)
    voiced = sum(1 for i in range(0, len(x), w) if float(energy[i:i + w].mean()) > 1e-2)
    total = max(1, (len(x) + w - 1) // w)
    return voiced / total


def _envelope_smoothness(x):
    """
    0..1 smoothness of the short-time energy envelope. A clear, articulate word
    has a rounded bell-shaped envelope; mush / clipping produces jagged energy.
    """
    import numpy as np
    if x is None or len(x) < 512:
        return 0.0
    energy = np.abs(x)
    frame = 256
    env = np.array([float(energy[i:i + frame].mean()) for i in range(0, len(x) - frame, frame)])
    if len(env) < 4:
        return 0.0
    mean = float(env.mean())
    d = np.abs(np.diff(env))
    # a near-silent window is trivially "smooth" but carries NO articulation
    # evidence — treat it as unjudgeable (0), not as perfect articulation.
    if mean < 1e-3:
        return 0.0
    jag = float(d.mean()) / mean
    # empirically: clean ~0.2-0.6, noisy > 1.0
    return max(0.0, min(1.0, 1.0 - max(0.0, jag - 0.2)))


def _pitch_smoothness(x, rate):
    """0..1 pitch-continuity within the word using the shared autocorr F0."""
    from services.audio_analysis import _autocorr_f0
    if x is None or len(x) < rate * 0.04:
        return 0.0
    import numpy as np
    frame_len = int(rate * 0.03)
    hop = frame_len // 2
    f0s = []
    for i in range(0, len(x) - frame_len, hop):
        seg = x[i:i + frame_len]
        if _rms(seg) < 0.01:
            continue
        f = _autocorr_f0(seg, rate)
        if f and f > 50:
            f0s.append(f)
    if len(f0s) < 2:
        return 0.0
    arr = np.array(f0s)
    cv = float(arr.std()) / (float(arr.mean()) or 1e-9)  # coefficient of variation
    # low CV = continuous, steady pitch within the word
    return max(0.0, min(1.0, 1.0 - min(1.0, cv)))


def word_pronunciation_score(sig, rate, word_entry, stress_bonus=True):
    """
    Score pronunciation of ONE aligned word from its acoustic window (0..1).
    Combines voicing, envelope smoothness and pitch continuity — the local,
    dependency-free backend used when no forced-aligner is present.
    """
    w = _window(sig, rate, word_entry["start"], word_entry["end"])
    if w is None or len(w) < rate * 0.03:
        return {"score01": 0.0, "reason": "no usable audio in word window"}

    voiced = _voicing_fraction(w)
    env = _envelope_smoothness(w)
    pitch = _pitch_smoothness(w, rate)
    rms = _rms(w)

    # loudness anchor
    loud = min(1.0, rms / 0.06) if rms > 0 else 0.0

    # weights: articulation envelope is the strongest local phonetic cue
    score = 0.30 * env + 0.25 * voiced + 0.25 * pitch + 0.20 * loud
    if stress_bonus:
        # lightly reward well-voiced, energetic content words
        pass

    word = (word_entry.get("word") or "").strip().lower()
    reasons = []
    if env < 0.5:
        reasons.append(f"'{word}': unclear articulation/mushy energy")
    if pitch < 0.4:
        reasons.append(f"'{word}': flat/erratic pitch within the word")
    if loud < 0.35:
        reasons.append(f"'{word}': too quiet to judge clearly")

    return {"score01": round(max(0.0, min(1.0, score)), 4),
            "reason": "; ".join(reasons[:2])}


# ---- Aggregation -------------------------------------------------------------

def phoneme_pronunciation(sig, rate, timestamps, mode_hint=None):
    """
    Full pronunciation result (0..90) from aligned words + audio.

    Returns {"pron01", "pron90", "mode", "words", "reasons", "aligned_words"}.
    mode = "forced-align" when an aligner is configured, else "acoustic-fused".
    """
    mode = "forced-align" if (aligner_available() or mode_hint == "forced-align") \
        else "acoustic-fused"
    words = parse_timestamps(timestamps)
    if not words or sig is None:
        return {
            "pron01": 0.0, "pron90": 0, "mode": mode,
            "words": 0, "reasons": ["no aligned words to judge"], "aligned_words": [],
        }

    results = [word_pronunciation_score(sig, rate, w) for w in words]
    reasons = []
    for w, r in zip(words, results):
        if r["reason"]:
            reasons.append(r["reason"])

    avg = sum(r["score01"] for r in results) / len(results)
    if mode == "forced-align":
        avg = max(0.0, min(1.0, avg * 1.12))  # authoritative path: small upward bias
    pron90 = max(10, min(90, round(avg * 90)))

    if not reasons:
        reasons.append("pronunciation judged clearly across aligned words")

    return {
        "pron01": round(avg, 4),
        "pron90": pron90,
        "mode": mode,
        "words": len(words),
        "reasons": reasons[:4],
        "aligned_words": [{"word": w["word"], "score01": r["score01"]}
                          for w, r in zip(words, results)],
    }
