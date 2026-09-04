"""
Tier 3c — Score equating to the fixed 10..90 ability ruler.

A raw task score is NOT a PTE score: it is a success fraction on one particular
item of one particular difficulty. Equating maps that item-level performance
onto the single, fixed latent ability ruler (10..90) so that:

  * the same raw performance on a HARD item equates HIGHER than on an EASY one,
  * the reported score means the same thing across every task/question type,
  * more items (multi-blank / multi-question tasks) tighten the estimate
    (larger n -> reduced standard error -> a more confident, less shrunk score).

Implementation reuses the IRT module (Tier 3b): we treat the observed success
fraction as a single virtual item outcome with difficulty b, estimate the
latent ability θ (MLE), shrink it toward the mean by the item-count precision,
then map θ onto the ruler. This gives a difficulty-aware, calibrated 10..90.
"""
import math

from services.irt import (
    ability_from_observations,
    difficulty_logit,
    question_difficulty_logit,
    ruler_map_degree,
)

RULER_MIN = 10
RULER_MAX = 90


def _as01(value):
    """Coerce any raw score into a 0..1 success fraction."""
    v = float(value or 0.0)
    return max(0.0, min(1.0, v / 90.0))


def _se_shrink(se, n_items):
    """Reduce the SE by sqrt(n) and return a confidence weight in 0..1."""
    if se is None:
        return 0.0
    return min(0.9, 0.3 + se * math.sqrt(max(1, n_items)))


def equate_score(score01, difficulty=None, n_items=1):
    """
    Equate a single-item success fraction onto the 10..90 ruler.
    Returns {"score": int, "difficulty_logit": float, "theta": float, "items": int}
    """
    s = _as01(score01 * 90)
    b = difficulty_logit(difficulty)
    n = max(1, int(n_items))

    est = ability_from_observations([{"b": b, "outcome": s}], theta0=0.0)

    # Item-count precision: with more items the same success is more credible,
    # so we pull the raw ruler value up toward the pure ability ruler (less
    # regression to the middle).
    raw_ruler = ruler_map_degree(est["theta"])
    precision = _se_shrink(est["se"], n)
    # Blend raw success toward the ability-anchored ruler by precision.
    effort = RULER_MIN + (RULER_MAX - RULER_MIN) * s
    blended = raw_ruler * precision + effort * (1.0 - precision)
    score = max(RULER_MIN, min(RULER_MAX, round(blended)))

    return {
        "score": score,
        "difficulty_logit": round(b, 3),
        "theta": est["theta"],
        "items": n,
    }


def equate_raw_to_90(raw90, difficulty=None, n_items=1):
    """Convenience: equate a raw 0..90 score onto the fixed ruler."""
    return equate_score(raw90 / 90.0, difficulty=difficulty, n_items=n_items)


def equate_traits(traits, difficulty=None, n_items=1):
    """
    Equate a spoken response's content/fluency/pronunciation (each 0..90) onto
    the fixed ruler and return the equated values plus a combined total.
    """
    eq = {}
    for key in ("content", "fluency", "pronunciation"):
        v = float(traits.get(key) or 0)
        eq[key] = equate_raw_to_90(v, difficulty=difficulty, n_items=n_items)["score"]

    # Primary-weight content, then fluency then pronunciation (PTE-style).
    total = round(0.55 * eq["content"] + 0.24 * eq["fluency"] + 0.21 * eq["pronunciation"])
    eq["score"] = max(RULER_MIN, min(RULER_MAX, total))
    return eq


def equate_question(question, traits_or_raw):
    """Equate using the difficulty read straight from a question dict."""
    b = question_difficulty_logit(question)
    n_items = 1
    content = (question or {}).get("content") or {}
    blanks = content.get("blanks") or {}
    n_items = max(1, len(blanks))
    if isinstance(traits_or_raw, dict):
        return equate_traits(traits_or_raw, difficulty=b, n_items=n_items)
    return equate_raw_to_90(float(traits_or_raw), difficulty=b, n_items=n_items)
