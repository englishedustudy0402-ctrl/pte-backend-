"""
Tier 3b — Item Response Theory (IRT) / question-difficulty model.

Pearson uses item-response style calibration: a candidate's reported score is an
estimate of their latent ability (theta, θ), not a naive percentage of correct
answers. An item's difficulty (b) and the person's ability (θ) combine through
the Rasch / 1-parameter logistic model:

    P(correct | θ, b) = 1 / (1 + e^-(θ - b))

This module provides:
  * a difficulty-labelled anchor (easy / medium / hard -> logit b),
  * the Item Characteristic Curve P,
  * a Maximum-Likelihood ability estimator (Newton-Raphson) that accepts whole
    or partial credit and never diverges to ±infinity (continuity-corrected),
  * a fixed θ -> 10..90 ability ruler (the equating bridge reused by Tier 3c),
  * IRT difficulty weighting so that correctly handling a HARD item earns more
    than correctly handling an easy one.

Because the difficulty anchors are ordinal labels from our own item bank, this
is a faithful *reproduction* of IRT scoring (the real PTE θ scale and item
parameters are proprietary and unknowable), not an exact copy.
"""
import math

MAX_ABS_THETA = 8.0       # clamp for the latent ability search space
RULER_MIN = 10            # eventual score ruler floor
RULER_MAX = 90            # eventual score ruler ceiling

# Ordinal difficulty labels -> logit difficulty anchors. These are calibrated
# so that a typical test-taker (θ=0) gets roughly: easy ≈ 82% P, medium ≈ 50% P,
# hard ≈ 18% P — a realistic spread across the item bank.
DIFFICULTY_LOGITS = {"easy": -1.5, "medium": 0.0, "hard": 1.5}
DEFAULT_DIFFICULTY = "medium"


# --------------------------------------------------------------------------
# Difficulty helpers
# --------------------------------------------------------------------------

def difficulty_logit(difficulty):
    """Map a question's difficulty label (or numeric value) to a logit b."""
    if difficulty is None:
        return DIFFICULTY_LOGITS[DEFAULT_DIFFICULTY]
    if isinstance(difficulty, (int, float)) and not isinstance(difficulty, bool):
        return float(difficulty)
    key = str(difficulty).strip().lower()
    return DIFFICULTY_LOGITS.get(key, DIFFICULTY_LOGITS[DEFAULT_DIFFICULTY])


def question_difficulty_logit(question):
    """Pull the difficulty parameter from a question dict (content/answer_data/root)."""
    for source in (
        (question or {}).get("answer_data") or {},
        (question or {}).get("content") or {},
        (question or {}),
    ):
        val = source.get("difficulty")
        if val is not None:
            return difficulty_logit(val)
    return difficulty_logit(None)


# --------------------------------------------------------------------------
# Item Characteristic Curve (Rasch / 1PL)
# --------------------------------------------------------------------------

def p_correct(theta, b):
    """P(correct | θ, b) under the 1-parameter logistic model, in 0..1."""
    return 1.0 / (1.0 + math.exp(-(_clamp_theta(theta) - b)))


def info(theta, b):
    """Fisher information of a single item at ability θ (max at θ = b)."""
    p = p_correct(theta, b)
    return p * (1.0 - p)


# --------------------------------------------------------------------------
# Ability estimation (MLE, Newton-Raphson)
# --------------------------------------------------------------------------

def _clamp_theta(theta):
    return max(-MAX_ABS_THETA, min(MAX_ABS_THETA, float(theta)))


def ability_from_observations(observations, theta0=0.0, iterations=60, eps=1e-9):
    """
    Estimate latent ability θ by maximum likelihood.

    observations: iterable of {"b": item difficulty logit, "outcome": 0..1}
                  (outcome may be partial credit, e.g. 0.6 for 60% correct).
    Returns: {"theta": MLE, "n": item count, "se": std err, "mean_outcome": 0..1}

    Newton-Raphson on the log-likelihood with a tiny continuity correction so
    that an all-correct or all-incorrect streak returns a finite θ (a real MLE
    would be ±∞; we clamp to the search box and report the boundary as-is).
    """
    b_list = [float(o["b"]) for o in observations]
    y_list = [float(min(1.0, max(0.0, o["outcome"]))) for o in observations]
    n = len(b_list)
    if n == 0:
        return {"theta": 0.0, "n": 0, "se": None, "mean_outcome": 0.0}

    theta = _clamp_theta(theta0)
    for _ in range(iterations):
        grad = 0.0
        hess = 0.0
        for b, y in zip(b_list, y_list):
            p = p_correct(theta, b)
            # continuity correction: treat perfect/missing outcomes as ~0.999/0.001
            grad += y - _clamped_p(y, p)
            hess -= info(theta, b)
        if hess == 0:
            break
        step = grad / hess
        new_theta = _clamp_theta(theta - step)
        if abs(new_theta - theta) < eps:
            theta = new_theta
            break
        theta = new_theta

    # Standard error from the information (iff we have enough items / spread).
    total_info = sum(info(theta, b) for b in b_list)
    se = (1.0 / math.sqrt(total_info)) if total_info > 1e-12 else None
    if se is not None and theta in (-MAX_ABS_THETA, MAX_ABS_THETA):
        se = None  # boundary estimate: SE unreliable

    return {
        "theta": round(theta, 5),
        "n": n,
        "se": round(se, 5) if se is not None else None,
        "mean_outcome": round(sum(y_list) / n, 5),
    }


def _clamped_p(y, p):
    """Continuity-corrected expected probability to keep grad finite at extremes."""
    if y >= 1.0:
        return 0.999
    if y <= 0.0:
        return 0.001
    return max(0.001, min(0.999, p))


# --------------------------------------------------------------------------
# θ -> 10..90 ability ruler (equating bridge; reused by Tier 3c)
# --------------------------------------------------------------------------

def ruler_map_degree(theta):
    """
    Map a θ ruler degree to the official-looking 10..90 scale.

    The ruler is anchored so that:
      θ = -4  -> 10  (very low ability)
      θ =  0  -> 50  (mid ability)
      θ = +4  -> 90  (very high ability)
    giving a smooth logistic curve PTE-style rather than a raw percentage.
    """
    t = _clamp_theta(theta)
    s = 1.0 / (1.0 + math.exp(-t * 0.75))
    return round(RULER_MIN + (RULER_MAX - RULER_MIN) * s)


def observations_to_ruler(observations, theta0=0.0):
    """Estimate θ from observations and return the equated 10..90 ability score."""
    est = ability_from_observations(observations, theta0=theta0)
    return est, ruler_map_degree(est["theta"])


# --------------------------------------------------------------------------
# IRT difficulty-weighted grading
# --------------------------------------------------------------------------

def difficulty_weight(score01, difficulty=None, theta0=0.0):
    """
    Re-weight a 0..1 raw success fraction by item difficulty so that correctly
    handling a hard item is worth more than an easy one.

    score01   : raw success fraction (0..1) for this item,
    difficulty: label ("easy"/"medium"/"hard") or numeric logit,
    theta0    : assumed ability anchor for the weighting curve.

    Returns an adjusted 0..1 fraction (never lower than the raw fraction).
    """
    s = max(0.0, min(1.0, float(score01)))
    b = difficulty_logit(difficulty)
    p0 = p_correct(theta0, b)          # expected success for an average taker
    # Scale: getting it right on a hard item (low p0) is rewarded more than
    # getting an easy item (high p0) right.
    if p0 <= 0 or p0 >= 1:
        return s
    w = p0 / (1.0 - p0)                # odds of success at the anchor
    # The adjusted success keeps the same "surprise" (log-odds ratio) as a
    # same-ability person earning it on an average item.
    odds = (s / (1.0 - s)) if s < 1 else 1e6
    adjusted = odds / (w + 1e-12)
    adj = adjusted / (1.0 + adjusted)
    return max(s, min(1.0, adj))
