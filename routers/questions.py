from fastapi import APIRouter, Depends, Query, HTTPException
from middleware.security import require_active_plan, require_authenticated, get_supabase
from typing import Optional, Literal
import json
import random
import uuid as uuidlib

router = APIRouter()

# Map frontend/offline task names to DB section names (authoritative lookup for
# calls that identify a question by task alone).
SECTION_MAP = {
    'read_aloud': 'speaking',
    'repeat_sentence': 'speaking',
    'answer_short': 'speaking',
    'describe_image': 'speaking',
    'retell_lecture': 'speaking',
    'respond_to_situation': 'speaking',
    'summarize_group_discussion': 'speaking',
    'summarize': 'writing',
    'essay': 'writing',
    'mcq': 'reading',
    'mcq_multi': 'reading',
    'reorder': 'reading',
    'fill_blanks': 'reading',
    'rw_fill_blanks': 'reading',
    'dictation': 'listening',
    'sst': 'listening',
    'hcs': 'listening',
    'missing_word': 'listening',
    'incorrect_words': 'listening',
    # legacy aliases accepted for backwards compatibility
    'fill_blanks_rw': 'reading',
    'summarize_spoken': 'listening',
    'listening_mcq': 'listening',
    'listening_fill': 'listening',
    'highlight_summary': 'listening',
    'highlight_incorrect': 'listening',
}

PREMIUM_TIER = "premium"
FREE_TIER = "free"


def _tier_for(profile) -> str:
    return PREMIUM_TIER if profile.get("plan") == "pro" else FREE_TIER


def _visible_tiers(profile) -> list:
    """Tiers this user may read. Non-pro users get the free subset only."""
    return [PREMIUM_TIER, FREE_TIER] if _tier_for(profile) == PREMIUM_TIER else [FREE_TIER]


def _base_query(profile):
    """Shared question selector: active + published + tier-gated."""
    return (
        get_supabase()
        .table("questions")
        .select("*")
        .eq("is_active", True)
        .eq("status", "published")
        .in_("tier", _visible_tiers(profile))
    )


def _apply_scope(query, section, task, exam, difficulty):
    if section:
        query = query.eq("section", section.lower())
    if task:
        query = query.eq("task", task)
    if exam:
        query = query.eq("exam", exam.lower())
    if difficulty:
        query = query.eq("difficulty", difficulty.lower())
    return query


def _scoped(profile, section, task, exam, difficulty):
    """Fresh tier-gated + scoped chain (never reuse a builder after execute)."""
    return _apply_scope(_base_query(profile), section, task, exam, difficulty)


@router.get("/")
async def get_questions(
    section: Optional[str] = None,
    task: Optional[str] = None,
    exam: Optional[str] = None,
    difficulty: Optional[Literal["easy", "medium", "hard"]] = None,
    limit: int = Query(default=5, le=20),
    profile=Depends(require_authenticated),
):
    """Metadata-only listing (id, question_key, section, task, difficulty,
    tier, exam). Content/answer_data are never returned in bulk; single-question
    delivery happens via /next or /{id}."""
    query = (
        get_supabase()
        .table("questions")
        .select("id,question_key,section,task,difficulty,tier,exam,status")
        .eq("is_active", True)
        .eq("status", "published")
        .in_("tier", _visible_tiers(profile))
    )
    query = _apply_scope(query, section, task, exam, difficulty)
    result = query.limit(limit).execute()
    return result.data


@router.get("/next")
async def next_question(
    section: str,
    task: Optional[str] = None,
    exam: Optional[str] = None,
    difficulty: Optional[Literal["easy", "medium", "hard"]] = None,
    history: Optional[str] = None,
    mode: Optional[Literal["practice", "mock"]] = Query(default="practice"),
    index: int = Query(default=1, ge=1, le=500),
    profile=Depends(require_active_plan),
):
    """Single-question delivery. `index` is the deterministic question_key that
    mirrors the offline bank id. In `mock` mode `index` is ignored and a random
    question matching the filters is served instead.

    Adaptive (Tier 4a): pass `history` as a JSON array of prior item outcomes,
    e.g. [{"difficulty":"medium","outcome":1}, ...]. An estimated ability (IRT)
    then overrides `difficulty` with the most informative next label, dosing
    harder after strong answers and easier after weak ones."""
    db_section = section.lower()

    if history:
        try:
            from services import adaptive as adaptive_svc
            parsed = json.loads(history)
            if isinstance(parsed, list) and parsed:
                sel = adaptive_svc.select_next(parsed, previous_label=difficulty)
                difficulty = sel["difficulty"]
        except (ValueError, TypeError):
            pass  # malformed history -> keep the requested difficulty
    if task and task in SECTION_MAP:
        db_section = SECTION_MAP[task]

    # Build a fresh query chain per lookup: supabase-py builders are mutated by
    # maybe_single()/execute(), so reusing one builder produces empty results.
    data = None
    if mode != "mock":
        try:
            res = _scoped(profile, db_section, task, exam, difficulty).eq("question_key", index).limit(1).execute()
            data = res.data[0] if res.data else None
        except Exception:
            data = None
    if not data:
        res2 = _scoped(profile, db_section, task, exam, difficulty).limit(1).execute()
        data = res2.data[0] if res2.data else None
    if not data:
        raise HTTPException(status_code=404, detail=f"No questions found for section={db_section} task={task}")
    return data


@router.get("/random")
async def get_random_question(
    section: str,
    task: Optional[str] = None,
    exam: Optional[str] = None,
    difficulty: Optional[Literal["easy", "medium", "hard"]] = None,
    profile=Depends(require_active_plan),
):
    """Single random question, tier-gated (premium bank only for pro)."""
    db_section = section.lower()
    if task and task in SECTION_MAP:
        db_section = SECTION_MAP[task]

    query = _base_query(profile)
    query = _apply_scope(query, db_section, task, exam, difficulty)

    result = query.execute()
    if not result.data:
        raise HTTPException(status_code=404, detail=f"No questions found for section={db_section} task={task}")
    return random.choice(result.data)


@router.get("/stats/me")
async def my_stats(profile=Depends(require_authenticated)):
    supabase = get_supabase()
    attempts = supabase.table("attempts")\
        .select("section,score,task,created_at")\
        .eq("user_id", profile["id"])\
        .order("created_at", desc=True)\
        .limit(200)\
        .execute()

    data = attempts.data
    if not data:
        return {"attempted": 0, "avg_score": 0, "by_section": {}}

    by_section: dict = {}
    for a in data:
        s = a.get("section", "unknown")
        if s not in by_section:
            by_section[s] = []
        if a.get("score"):
            by_section[s].append(a["score"])

    section_stats = {
        s: {
            "count": len(scores),
            "avg": round(sum(scores) / len(scores)) if scores else 0,
            "best": max(scores) if scores else 0,
        }
        for s, scores in by_section.items()
    }

    all_scores = [a["score"] for a in data if a.get("score")]
    return {
        "attempted": len(data),
        "avg_score": round(sum(all_scores) / len(all_scores)) if all_scores else 0,
        "by_section": section_stats,
    }


@router.get("/{question_id}")
async def get_question(
    question_id: str,
    section: Optional[str] = None,
    task: Optional[str] = None,
    profile=Depends(require_active_plan),
):
    """Single question by DB uuid or by offline address (section, task, index)."""
    supabase = get_supabase()
    query = _base_query(profile)

    try:
        parsed = uuidlib.UUID(question_id)
        is_uuid = True
    except (ValueError, AttributeError):
        parsed = None
        is_uuid = False

    if is_uuid:
        res = query.eq("id", str(parsed)).maybe_single().execute()
    else:
        if not question_id.isdigit():
            raise HTTPException(status_code=422, detail="question_id must be a uuid or an offline index")
        if not section or not task:
            raise HTTPException(status_code=422, detail="section and task are required for offline index lookups")
        res = (
            _apply_scope(query, SECTION_MAP.get(task, section), task, None, None)
            .eq("question_key", int(question_id))
            .maybe_single()
            .execute()
        )

    if not res.data:
        raise HTTPException(status_code=404, detail="Question not found")
    return res.data