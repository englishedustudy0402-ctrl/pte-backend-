"""
Tier 3e — Self-calibration from usage.

A skill estimate should get MORE trustworthy and MORE current as the user does
more tasks. This module adapts the raw skill average so that:

  * reliability  : each score is weighted by its IRT information (hard & easy
                   items, and greater total volume, tighten the estimate),
  * recency      : scores decay exponentially with age so improvement (or
                   decline) is reflected quickly — a 6-month-old 90 should not
                   prop up a present-day 50,
  * online       : a single fresh score can update a stored skill estimate in
                   place (running estimate) WITHOUT reprocessing all history.

Everything is pure/numpy-free and returns 0..90 values (floored at 10), so it
plugs straight into the skill ladder (Tier 3d) and the confidence band (4b).
"""
import math

SKILL_MIN = 10
SKILL_MAX = 90
DEFAULT_HALF_LIFE_DAYS = 30.0   # a skill "half-life": evidence halves every ~30 days


def _clamp(v):
    return float(max(SKILL_MIN, min(SKILL_MAX, v)))


def info_weight(b):
    """Normalised per-item precision: IRT information at the mean ability,
    divided by its max (0.25) so the result is in 0..1. Extreme-difficulty and
    well-matched items are most informative; a fully trivial item carries ~0."""
    from services.irt import info
    return min(1.0, info(0.0, b) / 0.25)


def recency_weight(age_days, half_life_days=DEFAULT_HALF_LIFE_DAYS):
    """Exponential time-decay of evidence, 0..1. age 0 -> 1.0; age == half-life
    -> 0.5; older evidence fades toward 0."""
    if age_days is None or age_days < 0:
        return 1.0
    return math.pow(0.5, age_days / max(1e-6, half_life_days))


def confidence(n_observations):
    """0..1 confidence from observation count (more data -> more confidence).
    Plateaus toward 1.0 as n grows; used for the margin-of-error band."""
    n = max(0, int(n_observations))
    if n == 0:
        return 0.0
    return min(1.0, 1.0 - math.exp(-n / 8.0))


def estimate(scores):
    """
    Reliability + recency aware estimate of ONE skill from raw scores.

    scores: list of {"score": 0..90, "difficulty_b": float (logit) | None,
                     "age_days": float | None}
    Returns {"value": 0..90, "weight": total reliability weight,
             "n": count, "confidence": 0..1}
    """
    if not scores:
        return {"value": SKILL_MIN, "weight": 0.0, "n": 0, "confidence": 0.0}

    from services.irt import difficulty_logit

    value_sum = 0.0
    weight_sum = 0.0
    n = 0
    for s in scores:
        sc = _clamp(float(s.get("score") or 0))
        b = difficulty_logit(s.get("difficulty_b"))
        rel = info_weight(b)                          # item precision 0..1
        rec = recency_weight(s.get("age_days"))       # recency 0..1
        w = max(0.05, rel * rec)                      # floor so 1 item counts a little
        value_sum += sc * w
        weight_sum += w
        n += 1

    value = _clamp(value_sum / weight_sum if weight_sum > 0 else 0.0)
    return {
        "value": value,
        "weight": round(weight_sum, 3),
        "n": n,
        "confidence": round(confidence(n), 3),
    }


def online_update(current, new_score, opts=None):
    """
    Update a stored skill estimate with ONE fresh score, without reprocessing
    history.

    current: {"value": 0..90, "weight": w, "n": n, ...} (from estimate())
    new_score: {"score": 0..90, "difficulty_b": | None, "age_days": 0, ...}
    opts: {"half_life_days": float}

    Returns an updated estimate dict (same shape as estimate()).
    """
    opts = opts or {}
    half = float(opts.get("half_life_days", DEFAULT_HALF_LIFE_DAYS))
    cur_value = _clamp(float((current or {}).get("value") or SKILL_MIN))
    cur_weight = max(0.0, float((current or {}).get("weight") or 0.0))
    cur_n = max(0, int((current or {}).get("n") or 0))

    from services.irt import difficulty_logit
    b = difficulty_logit(new_score.get("difficulty_b"))
    new_val = _clamp(float(new_score.get("score") or 0))

    # decay old evidence by time since the last update, then fold in the new
    age = float(new_score.get("age_days") or 0.0)
    decayed_weight = cur_weight * recency_weight(age, half)

    new_weight = max(0.05, info_weight(b))
    # tiny mixing term so a single well-answered hard item nudges the value up
    mix = info_weight(b)
    weighted_new = new_val * mix + cur_value * (1.0 - mix)

    value_sum = cur_value * decayed_weight + weighted_new * new_weight
    weight_sum = decayed_weight + new_weight
    value = _clamp(value_sum / weight_sum if weight_sum > 0 else cur_value)

    return {
        "value": value,
        "weight": round(weight_sum, 3),
        "n": cur_n + 1,
        "confidence": round(confidence(cur_n + 1), 3),
    }
