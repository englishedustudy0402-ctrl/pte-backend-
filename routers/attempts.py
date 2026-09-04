from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional, Literal
from middleware.security import require_active_plan, require_authenticated, get_supabase
from routers.sessions import _resolve_question
from services import scoring_engine
from services.scoring_engine import grade, persist
from services.ai_gateway import commentary

router = APIRouter()


class CreateAttemptRequest(BaseModel):
    question_id: Optional[str] = None
    section: Optional[str] = None
    task: Optional[str] = None
    question_key: Optional[int] = None
    mode: Optional[Literal["practice", "sectional", "mock", "exam"]] = "practice"
    audio_path: Optional[str] = None
    duration_ms: Optional[int] = None


class SubmitAttemptRequest(BaseModel):
    answer: Optional[str] = ""
    answer_text: Optional[str] = None
    was_timed_out: Optional[bool] = False
    time_taken: Optional[float] = None
    tab_penalties: Optional[int] = 0


@router.post("/")
async def create_attempt(body: CreateAttemptRequest, profile=Depends(require_active_plan)):
    """Create an attempt owned by the authenticated user. The body never
    carries a user_id — ownership always comes from the JWT."""
    supabase = get_supabase()
    question, err = _resolve_question(supabase, body, profile)
    if err == 422:
        raise HTTPException(status_code=422, detail="Supply question_id (uuid/offline index) or section+task+question_key")
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")

    row = {
        "user_id": profile["id"],
        "question_id": str(question["id"]),
        "section": question["section"],
        "task": question["task"],
        "status": "in_progress",
    }
    if body.audio_path:
        if not body.audio_path.startswith(f"{profile['id']}/") or ".." in body.audio_path:
            raise HTTPException(status_code=400, detail="Invalid audio path")
        row["audio_path"] = body.audio_path
    if body.duration_ms is not None:
        row["duration_ms"] = body.duration_ms

    res = supabase.table("attempts").insert(row).execute()
    attempt = res.data[0]
    return {"attempt_id": attempt["id"], "status": attempt["status"], "question": question}


@router.get("/my")
async def my_attempts(limit: int = 20, profile=Depends(require_authenticated)):
    """Recent attempts (with scores) for the authenticated user."""
    supabase = get_supabase()
    res = (
        supabase.table("attempts")
        .select("id,question_id,section,task,score,status,answer_text,audio_path,created_at")
        .eq("user_id", profile["id"])
        .order("created_at", desc=True)
        .limit(max(1, min(100, limit)))
        .execute()
    )
    return res.data


@router.get("/{attempt_id}")
async def get_attempt(attempt_id: str, profile=Depends(require_authenticated)):
    """Own attempt detail, including the question content and any score record."""
    supabase = get_supabase()
    res = (
        supabase.table("attempts")
        .select("*")
        .eq("id", attempt_id)
        .eq("user_id", profile["id"])
        .limit(1)
        .execute()
    )
    if not (res and res.data):
        raise HTTPException(status_code=404, detail="Attempt not found")
    attempt = res.data[0]

    question = {}
    if attempt.get("question_id"):
        q = (
            supabase.table("questions")
            .select("id,section,task,difficulty,question_text,content")
            .eq("id", attempt["question_id"])
            .limit(1)
            .execute()
        )
        question = q.data[0] if q and q.data else {}

    score = {}
    s = supabase.table("scores").select("*").eq("attempt_id", attempt_id).maybe_single().execute()
    if s and s.data:
        comp = supabase.table("score_components").select("*").eq("score_id", s.data["id"]).execute()
        score = {**s.data, "components": comp.data or []}

    return {"attempt": attempt, "question": question, "score": score}


@router.post("/{attempt_id}/submit")
async def submit_attempt(attempt_id: str, body: SubmitAttemptRequest, profile=Depends(require_active_plan)):
    """Submit + canonical deterministic scoring. Duplicate submission -> 409."""
    supabase = get_supabase()
    res = (
        supabase.table("attempts")
        .select("*")
        .eq("id", attempt_id)
        .eq("user_id", profile["id"])
        .limit(1)
        .execute()
    )
    if not (res and res.data):
        raise HTTPException(status_code=404, detail="Attempt not found")
    attempt = res.data[0]
    if attempt["status"] != "in_progress":
        raise HTTPException(status_code=409, detail="Attempt already submitted")
    dup = supabase.table("scores").select("id").eq("attempt_id", attempt_id).limit(1).execute()
    if dup.data:
        raise HTTPException(status_code=409, detail="Attempt already scored")

    update = {
        "answer": body.answer or "",
        "answer_text": body.answer_text or (body.answer or ""),
        "was_timed_out": bool(body.was_timed_out),
        "tab_penalties": max(0, min(int(body.tab_penalties or 0), 50)),
        "status": "submitted",
    }
    if body.time_taken is not None:
        update["time_taken"] = max(0.0, min(float(body.time_taken), 4 * 3600))
    supabase.table("attempts").update(update).eq("id", attempt_id).execute()

    fresh = supabase.table("attempts").select("*").eq("id", attempt_id).limit(1).execute()
    attempt = fresh.data[0] if fresh and fresh.data else attempt

    question = (
        supabase.table("questions")
        .select("*")
        .eq("id", attempt["question_id"])
        .limit(1)
        .execute()
    )
    if not (question and question.data):
        raise HTTPException(status_code=404, detail="Question missing")
    question = question.data[0]

    result = grade(question, attempt, body.model_dump())
    persist(supabase, attempt, question, result)

    return {
        "attempt_id": attempt["id"],
        "score": result["overall_score"],
        "engine": result["engine"],
        "engine_version": result["version"],
        "components": result["components"],
        "meta": result["meta"],
    }


@router.post("/{attempt_id}/feedback")
async def attempt_feedback(attempt_id: str, profile=Depends(require_authenticated)):
    """Optional AI examiner commentary layered over the deterministic score.
    Never changes the score; returns prose feedback only."""
    supabase = get_supabase()
    res = (
        supabase.table("attempts")
        .select("*")
        .eq("id", attempt_id)
        .eq("user_id", profile["id"])
        .limit(1)
        .execute()
    )
    if not (res and res.data):
        raise HTTPException(status_code=404, detail="Attempt not found")
    attempt = res.data[0]
    if attempt["status"] != "submitted":
        raise HTTPException(status_code=409, detail="Attempt not submitted yet")

    question = (
        supabase.table("questions")
        .select("question_text,content,answer_data")
        .eq("id", attempt["question_id"])
        .limit(1)
        .execute()
    )
    q = question.data[0] if question and question.data else {}
    ref = ""
    qd = q.get("answer_data") or {}
    content = q.get("content") or {}
    ref = qd.get("model_answer") or qd.get("reference_text") or content.get("modelAnswer") or ""

    doc, engine = commentary(
        task=attempt["task"],
        question_text=q.get("question_text") or "",
        reference=ref,
        student_response=attempt.get("answer_text") or attempt.get("answer") or "",
        note=json_dumps(attempt.get("score")),
    )
    if not doc:
        return {
            "feedback": "AI commentary unavailable right now - deterministic score stands.",
            "tip": "",
            "encouragement": "",
            "ai_engine": "offline",
        }
    return {"ai_engine": engine, "score": attempt.get("score"), **doc}


def json_dumps(value):
    import json
    return json.dumps({"score": value}, default=str)