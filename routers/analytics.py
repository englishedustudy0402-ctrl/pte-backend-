from fastapi import APIRouter, Depends, Query
from middleware.security import require_authenticated, get_supabase

router = APIRouter()

# Analytics read endpoints. Everything is filtered to the authenticated user's
# own data (profile["id"]), i.e. IDOR-safe by construction. Aggregates come from
# public.user_statistics, which is rolled up by services/scoring_engine.persist
# on every scored attempt, plus recent rows from public.attempts.

SECTION_LABEL = {
    "speaking": "Speaking",
    "writing": "Writing",
    "reading": "Reading",
    "listening": "Listening",
}


@router.get("/overview")
async def analytics_overview(profile=Depends(require_authenticated)):
    """Dashboard-level aggregates for the current user."""
    supabase = get_supabase()

    recent = (
        supabase.table("attempts")
        .select("id,score,section,task,created_at,status")
        .eq("user_id", profile["id"])
        .order("created_at", desc=True)
        .limit(500)
        .execute()
    )
    rows = recent.data or []
    scored = [r for r in rows if r.get("score") is not None]
    total = len(rows)

    by_section: dict = {}
    for r in rows:
        sec = r.get("section") or "unknown"
        bucket = by_section.setdefault(sec, {"count": 0, "scores": []})
        bucket["count"] += 1
        if r.get("score") is not None:
            bucket["scores"].append(r["score"])
    section_stats = {
        sec: {
            "label": SECTION_LABEL.get(sec, sec),
            "count": b["count"],
            "avg": round(sum(b["scores"]) / len(b["scores"]), 1) if b["scores"] else 0,
            "best": max(b["scores"]) if b["scores"] else 0,
        }
        for sec, b in by_section.items()
    }

    tasks = (
        supabase.table("user_statistics")
        .select("*")
        .eq("user_id", profile["id"])
        .order("average_score", desc=True)
        .execute()
    )
    task_rows = tasks.data or []

    all_scores = [r["score"] for r in scored]
    return {
        "total_attempts": total,
        "scored_attempts": len(scored),
        "avg_score": round(sum(all_scores) / len(all_scores), 1) if all_scores else 0,
        "best_score": max(all_scores) if all_scores else 0,
        "by_section": section_stats,
        "by_task": [
            {
                "task_type": t["task_type"],
                "attempts": t.get("attempts") or 0,
                "average_score": round(t.get("average_score") or 0, 1),
                "best_score": t.get("best_score") or 0,
                "recent_score": t.get("recent_score") or 0,
                "updated_at": t.get("updated_at"),
            }
            for t in task_rows
        ],
    }


@router.get("/skills")
async def analytics_skills(profile=Depends(require_authenticated)):
    """Per-task skill scores from the persisted rollups (Skills Overview data)."""
    supabase = get_supabase()
    tasks = (
        supabase.table("user_statistics")
        .select("task_type,attempts,average_score,best_score,recent_score,updated_at")
        .eq("user_id", profile["id"])
        .order("average_score", desc=True)
        .execute()
    )
    return (tasks.data or []) or []


TASK_COMMUNICATIVE = {
    "read_aloud": "speaking", "repeat_sentence": "speaking",
    "describe_image": "speaking", "retell_lecture": "speaking",
    "answer_short": "speaking",
    "summarize": "writing", "essay": "writing",
    "mcq": "reading", "mcq_multi": "reading", "reorder": "reading",
    "fill_blanks": "reading", "fill_blanks_rw": "reading",
    "summarize_spoken": "listening", "listening_mcq": "listening",
    "listening_fill": "listening", "highlight_summary": "listening",
    "missing_word": "listening", "highlight_incorrect": "listening",
    "dictation": "listening",
}
ENABLING_KEYS = ("grammar", "vocabulary", "pronunciation", "oral fluency",
                 "spelling", "written discourse")


@router.get("/calibrated")
async def analytics_calibrated(profile=Depends(require_authenticated)):
    """Razor-crisp calibrated skill ladder (Tier 3e) from the user's persisted
    rollups. Every skill is reliability-weighted (IRT info) and recency-decayed,
    returns 0..90 values floored at 10 plus a 0..1 confidence per skill.
    IDOR-safe: only the authenticated user's own rollups are read."""
    from services.calibration import estimate
    from services.irt import difficulty_logit
    from services.confidence import skill_band, overall_band
    from datetime import datetime, timezone

    supabase = get_supabase()
    rows = (
        supabase.table("user_statistics")
        .select("task_type,average_score,attempts,updated_at")
        .eq("user_id", profile["id"])
        .execute()
    )
    data = rows.data or []
    if not data:
        return {
            "communicative": {k: {"value": 10, "confidence": 0, "n": 0} for k in
                              ("listening", "reading", "speaking", "writing")},
            "enabling": {k: {"value": 10, "confidence": 0, "n": 0} for k in ENABLING_KEYS},
            "source": {},
        }

    now = datetime.now(timezone.utc)

    def age_days(updated_at):
        try:
            dt = datetime.fromisoformat(str(updated_at).replace("Z", "+00:00"))
            return max(0.0, (now - dt).total_seconds() / 86400.0)
        except (ValueError, TypeError):
            return 0.0

    # collect per-task evidence (difficulty from label heuristic)
    comm_scores = {k: [] for k in ("listening", "reading", "speaking", "writing")}
    enabling_scores = {k: [] for k in ENABLING_KEYS}
    source = {}
    # enabling-trait attribution (mirrors frontend skillLadder)
    enabling_traits = {
        "read_aloud": [("pronunciation", 1.0), ("oral fluency", 0.6), ("vocabulary", 0.15)],
        "repeat_sentence": [("pronunciation", 1.0), ("oral fluency", 0.9),
                            ("grammar", 0.3), ("vocabulary", 0.25)],
        "describe_image": [("oral fluency", 0.8), ("pronunciation", 0.5),
                           ("vocabulary", 0.7), ("grammar", 0.5)],
        "retell_lecture": [("oral fluency", 0.8), ("pronunciation", 0.5),
                           ("vocabulary", 0.8), ("grammar", 0.6)],
        "answer_short": [("vocabulary", 0.5), ("pronunciation", 0.3)],
        "summarize": [("grammar", 1.0), ("vocabulary", 1.0), ("spelling", 0.7),
                      ("written discourse", 1.0)],
        "essay": [("grammar", 1.0), ("vocabulary", 1.0), ("spelling", 0.9),
                  ("written discourse", 1.0)],
        "fill_blanks_rw": [("vocabulary", 0.9), ("grammar", 0.8)],
        "fill_blanks": [("vocabulary", 0.9), ("grammar", 0.4)],
        "reorder": [("grammar", 0.9), ("vocabulary", 0.3)],
        "mcq": [("vocabulary", 0.6)],
        "mcq_multi": [("vocabulary", 0.6)],
        "dictation": [("spelling", 1.0), ("vocabulary", 0.7)],
        "listening_fill": [("spelling", 0.8), ("vocabulary", 0.6)],
        "missing_word": [("spelling", 0.7), ("vocabulary", 0.6)],
        "listening_mcq": [("vocabulary", 0.5)],
        "highlight_summary": [("vocabulary", 0.4)],
        "highlight_incorrect": [("vocabulary", 0.5)],
        "summarize_spoken": [("grammar", 0.7), ("vocabulary", 0.8),
                             ("spelling", 0.5), ("written discourse", 0.8)],
    }

    for r in data:
        task = r.get("task_type")
        avg = r.get("average_score")
        if not task or avg is None:
            continue
        try:
            avg = float(avg)
        except (TypeError, ValueError):
            continue
        ad = age_days(r.get("updated_at"))
        b = difficulty_logit(task)  # label from task_name; fallback medium
        score = max(10.0, min(90.0, avg))
        source[task] = {"score": round(score, 1), "attempts": r.get("attempts") or 0}

        comm = TASK_COMMUNICATIVE.get(task)
        if comm:
            comm_scores[comm].append({"score": score, "difficulty_b": b, "age_days": ad})
        for skill, w in enabling_traits.get(task, []):
            enabling_scores[skill].append({"score": score, "difficulty_b": b, "age_days": ad})

    def est_map(d):
        out = {}
        for k, scores in d.items():
            e = estimate(scores)
            band = skill_band(e["value"], e["n"])
            out[k] = {
                "value": round(e["value"], 1),
                "confidence": e["confidence"],
                "n": e["n"],
                "min": band["min"],
                "max": band["max"],
                "width": band["width"],
            }
        return out

    comm = est_map(comm_scores)
    enab = est_map(enabling_scores)

    # overall band from the four communicative skill bands
    overall_value = round(
        sum(comm[k]["value"] for k in ("listening", "reading", "speaking", "writing")) / 4.0, 1
    )
    overall = overall_band(overall_value, [comm[k] for k in ("listening", "reading", "speaking", "writing")])
    overall_out = {
        "value": overall_value,
        "min": overall["min"],
        "max": overall["max"],
        "width": overall["width"],
        "confidence": overall["confidence"],
        "n": sum(comm[k]["n"] for k in ("listening", "reading", "speaking", "writing")),
    }

    return {
        "communicative": comm,
        "enabling": enab,
        "overall": overall_out,
        "source": source,
    }


@router.get("/trend")
async def analytics_trend(
    limit: int = Query(default=20, ge=1, le=100),
    profile=Depends(require_authenticated),
):
    """Recent score history (attempts, oldest -> newest) for simple charts."""
    supabase = get_supabase()
    rows = (
        supabase.table("attempts")
        .select("score,section,task,created_at")
        .eq("user_id", profile["id"])
        .eq("status", "submitted")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    series = [r for r in (rows.data or []) if r.get("score") is not None]
    series.reverse()
    return {
        "points": series,
        "labels": [p["created_at"] for p in series],
        "scores": [p["score"] for p in series],
    }