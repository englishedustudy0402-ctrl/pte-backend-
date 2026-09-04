"""
Tier 4a — Adaptive difficulty.

A real adaptive test does not serve a fixed difficulty: after a strong answer it
serves a HARDER item, after a weak one an EASIER item, converging on the item
difficulty where the test-taker succeeds ~50% of the time — the most informative
questions for estimating their true ability.

Under the Rasch/1PL model (Tier 3b) that optimal next item difficulty is simply
the current ability estimate θ (item info is maximised when b == θ). So the
selector:

  1. estimates θ from recent outcomes (full-history MLE, or one-step online),
  2. maps θ onto the question bank's ordinal labels (easy/medium/hard) with
     hysteresis so a single lucky/missed item does not flip the level,
  3. returns the label so the /next endpoint can serve a matching question.

The labels are returned with the computed θ so callers can also bias by demand
(e.g. starting ability, remaining test length).
"""
from services.irt import ability_from_observations, difficulty_logit

# threshold points on θ at which the served difficulty label steps up
_EASY_EDGE = -0.75   # θ below this -> easy
_HARD_EDGE = 0.75    # θ above this -> hard
# hysteresis dead-zone width (prevents flip-flopping on a borderline run)
_HYST = 0.35

ORDER = ["easy", "medium", "hard"]


def theta_from_history(history, theta0=0.0):
    """
    Estimate ability from a history of item outcomes.
    history: list of {"difficulty": label|logit, "outcome": 0..1}
    Returns the IRT ability record {theta, n, se, mean_outcome}.
    """
    obs = []
    for h in history:
        b = difficulty_logit(h.get("difficulty"))
        out = float(h.get("outcome") or 0)
        obs.append({"b": b, "outcome": max(0.0, min(1.0, out))})
    return ability_from_observations(obs, theta0=theta0)


def band_label(theta):
    """Map estimated θ -> easy|medium|hard band (no hysteresis, pure split)."""
    t = float(theta)
    if t < _EASY_EDGE:
        return "easy"
    if t > _HARD_EDGE:
        return "hard"
    return "medium"


def next_label(theta, previous_label=None):
    """
    Choose the next difficulty label for estimated ability θ, with hysteresis
    relative to the previously served label (if any) so a single item cannot
    bounce the level repeatedly.

    To CHANGE band you must cross the current band's own edge by the hysteresis
    margin; within the current band (plus its margin) you stay put. A candidate
    confidently inside a band is therefore served that (appropriately
    challenging) band rather than being held back or pushed up prematurely.
    """
    t = float(theta)
    if previous_label:
        prev = previous_label if previous_label in ORDER else "medium"
        if prev == "easy":
            # leave easy only when clearly above the easy side of medium
            return "medium" if t > _EASY_EDGE + _HYST else "easy"
        if prev == "hard":
            # leave hard only when clearly below the hard side of medium
            return "medium" if t < _HARD_EDGE - _HYST else "hard"
        # medium: leave up/down only when clearly past both edges
        if t > _HARD_EDGE + _HYST:
            return "hard"
        if t < _EASY_EDGE - _HYST:
            return "easy"
        return "medium"
    return band_label(t)


def select_next(history, previous_label=None, theta0=0.0):
    """
    Full-history selector: estimate ability from all prior outcomes and return
    the next difficulty label plus the underlying ability estimate.
    """
    est = theta_from_history(history, theta0=theta0)
    label = next_label(est["theta"], previous_label)
    return {"difficulty": label, "theta": est["theta"], "n": est["n"], "se": est["se"]}


def select_next_online(current_theta, outcome, difficulty, previous_label=None):
    """
    One-step online selector: given the current θ estimate and a single new
    outcome, update θ (Newton step) and return the next difficulty.

    current_theta  : current ability θ (float),
    outcome        : 0..1 success for the item just done,
    difficulty     : label/logit of that just-done item,
    previous_label : label served for the just-done item (for hysteresis).
    Returns {"difficulty": label, "theta": new_theta, "updated_from": ...}
    """
    b = difficulty_logit(difficulty)
    # one Newton–Raphson update of the log-likelihood at the current θ
    from services.irt import p_correct
    p = p_correct(current_theta, b)
    info = p * (1.0 - p)
    clamp = 8.0
    t = max(-clamp, min(clamp, float(current_theta)))
    if info > 1e-9:
        grad = float(outcome) - p
        t = max(-clamp, min(clamp, t + grad / info))
    label = next_label(t, previous_label)
    return {"difficulty": label, "theta": round(t, 5)}
