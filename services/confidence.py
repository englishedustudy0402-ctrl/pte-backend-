"""
Tier 4b — Margin of error / confidence band.

Real PTE reports every skill with an implicit confidence band: a score is not a
point, it is a range (e.g. "64 ± 7") reflecting measurement uncertainty. The
band depends on how much evidence (how many scored items) supports the skill and
on the IRT standard error of the ability estimate.

This module turns that into concrete numbers a scorecard can render:

  * sem(n, sigma)        -> standard error of the mean for n observations,
  * points_band(sem)     -> the ±band in PTE points (widens as evidence thins),
  * skill_band(...)      -> {min, max, width, confidence} for a single skill,
  * overall_band(...)    -> combined band for the overall score,
  * narrow(...)          -> regression-to-the-mean shrink for low confidence.

Everything is pure & numpy-free and returns 0..90 (floored at 10 / capped 90).
"""
import math

SKILL_MIN = 10
SKILL_MAX = 90
# score-population SD assumption for the score metric (0..90). Only used to
# scale the SEM; does not need to be exact to give a believable band.
POP_SD = 12.0
# how many PTE points a "full" SEM translates into (a tunable ruler anchor)
SEM_TO_POINTS = 6.0


def _clamp(v):
    return float(max(SKILL_MIN, min(SKILL_MAX, v)))


def sem(n, sigma=POP_SD):
    """Standard error of the mean for n independent observations."""
    n = max(1, int(n))
    return sigma / math.sqrt(n)


def points_band(sem_value):
    """Convert a standard error to a ± band in PTE points (clamped 0..15)."""
    return min(15.0, max(0.5, round(sem_value * SEM_TO_POINTS, 1)))


def skill_band(value, n, se_theta=None):
    """
    Confidence band for one skill value (0..90).

    value    : the reported skill score (0..90),
    n        : number of scored observations backing it,
    se_theta : optional IRT standard error of the ability estimate (θ units).
               When provided, it adds measurement error beyond sampling noise.

    Returns {"min","max","width","confidence"} all 0..90.
    """
    v = _clamp(value)
    n = max(1, int(n))

    # observation-derived precision
    sampling_sem = sem(n)
    # IRT ability SE (θ units ~ roughly half a PTE point each) folded in
    irt_pts = (float(se_theta) * 2.0) if se_theta else 0.0
    total_sem = math.sqrt(sampling_sem ** 2 + (irt_pts / SEM_TO_POINTS) ** 2)

    width = points_band(total_sem)
    lo = _clamp(v - width)
    hi = _clamp(v + width)

    confidence = 1.0 - math.exp(-n / 8.0)
    return {
        "min": round(lo, 1),
        "max": round(hi, 1),
        "width": round(width, 1),
        "confidence": round(min(1.0, confidence), 3),
        "n": n,
    }


def overall_band(overall, skill_bands):
    """
    Any single skill band can mislead; combining all the skills' widths gives a
    more honest overall band (root-sum-of-squares of the component bands).

    skill_bands : iterable of {"width": float} from skill_band().
    """
    widths = [abs(float(b.get("width") or 0)) for b in skill_bands if b]
    if not widths:
        return {"min": SKILL_MIN, "max": SKILL_MAX, "width": 90.0, "confidence": 0.0}
    combined = math.sqrt(sum(w * w for w in widths)) / math.sqrt(len(widths))
    lo = _clamp(float(overall) - combined)
    hi = _clamp(float(overall) + combined)
    return {
        "min": round(lo, 1),
        "max": round(hi, 1),
        "width": round(combined, 1),
        "confidence": round(1.0 - math.exp(-len(widths) / 8.0), 3),
    }


def narrow(value, confidence01):
    """
    Reliability shrink: pull a low-confidence score toward the mid ruler (50) so
    a barely-measured skill does not look falsely extreme. High-confidence
    scores are left essentially unchanged.
    """
    v = _clamp(value)
    c = max(0.0, min(1.0, float(confidence01)))
    if c <= 0:
        return SKILL_MIN
    target = 50.0
    return round(_clamp(target + (v - target) * c), 1)
