from fastapi import APIRouter, Depends, HTTPException, Request, BackgroundTasks
from middleware.security import get_current_user, require_active_plan, log_audit, get_client_ip, get_supabase
from routers.questions import SECTION_MAP, _visible_tiers
from datetime import datetime, timezone
from pydantic import BaseModel
from typing import Optional
import uuid as uuidlib

router = APIRouter()

class StartSessionRequest(BaseModel):
    # Address a question either by DB uuid (question_id) or by the offline key
    # (section, task, question_key=index in 1..500). A digit-only question_id is
    # treated as an offline index.
    question_id: Optional[str] = None
    section: Optional[str] = None
    task: Optional[str] = None
    question_key: Optional[int] = None

class SubmitAnswerRequest(BaseModel):
    answer: str


def _resolve_question(supabase, body, profile):
    """Resolve the body to an active, published, tier-visible questions row.
    Returns (row or None, error_code or None)."""
    supabase = get_supabase()
    query = (
        supabase
        .table("questions")
        .select("*")
        .eq("is_active", True)
        .eq("status", "published")
        .in_("tier", _visible_tiers(profile))
    )

    if body.question_id:
        val = str(body.question_id).strip()
        try:
            as_uuid = uuidlib.UUID(val)
            res = query.eq("id", str(as_uuid)).maybe_single().execute()
            return (res.data, None) if res and res.data else (None, 404)
        except ValueError:
            if not val.isdigit():
                return None, 422
            idx = int(val)
    elif body.question_key is not None:
        idx = int(body.question_key)
    else:
        return None, 422

    section = (body.section or "").strip().lower() or SECTION_MAP.get(body.task or "")
    if not section or not body.task:
        return None, 422
    res = (
        query
        .eq("section", section)
        .eq("task", body.task)
        .eq("question_key", idx)
        .maybe_single()
        .execute()
    )
    return (res.data, None) if res and res.data else (None, 404)

@router.post("/start")
async def start_session(body: StartSessionRequest, request: Request, profile=Depends(require_active_plan)):
    supabase = get_supabase()
    ip = get_client_ip(request)

    question, err = _resolve_question(supabase, body, profile)
    if err == 422:
        raise HTTPException(status_code=422, detail="Supply question_id (uuid/offline index) or section+task+question_key")
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")

    question_id = str(question["id"])
    time_limit = question["content"].get("timeLimit", 120)

    # Check for existing session
    existing = supabase.table("exam_sessions")\
        .select("*")\
        .eq("user_id", profile["id"])\
        .eq("question_id", question_id)\
        .order("started_at", desc=True)\
        .limit(1)\
        .execute()

    if existing.data:
        sess = existing.data[0]
        if not sess.get("submitted_at") and not sess.get("is_expired"):
            elapsed = (datetime.now(timezone.utc) - _parse_dt(sess["started_at"])).total_seconds()
            remaining = max(0, time_limit - int(elapsed))
            if remaining > 0:
                return {"session_id": sess["id"], "time_remaining": remaining, "question": question}
            else:
                supabase.table("exam_sessions").update({"is_expired": True}).eq("id", sess["id"]).execute()

    # Create new session
    try:
        session = supabase.table("exam_sessions").insert({
            "user_id": profile["id"],
            "question_id": question_id,
            "section": question["section"],
            "task": question["task"],
            "time_limit_secs": time_limit,
            "ip_address": ip,
        }).execute()
        session_id = session.data[0]["id"]
    except Exception:
        # Fallback: reuse most recent session
        fallback = supabase.table("exam_sessions")\
            .select("*")\
            .eq("user_id", profile["id"])\
            .eq("question_id", question_id)\
            .order("started_at", desc=True)\
            .limit(1)\
            .execute()
        if not fallback.data:
            raise HTTPException(status_code=500, detail="Session error")
        session_id = fallback.data[0]["id"]

    log_audit(profile["id"], "session_started", {"question_id": question_id}, ip)
    return {"session_id": session_id, "time_remaining": time_limit, "question": question}


@router.post("/{session_id}/submit")
async def submit_answer(
    session_id: str,
    body: SubmitAnswerRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    profile=Depends(require_active_plan),
):
    supabase = get_supabase()
    ip = get_client_ip(request)

    sess_res = supabase.table("exam_sessions")\
        .select("*")\
        .eq("id", session_id)\
        .eq("user_id", profile["id"])\
        .single()\
        .execute()

    if not sess_res.data:
        raise HTTPException(status_code=404, detail="Session not found")

    sess = sess_res.data
    if sess.get("submitted_at"):
        raise HTTPException(status_code=409, detail="Already submitted")

    if sess.get("is_expired"):
        raise HTTPException(status_code=410, detail="Session expired")

    started_at = _parse_dt(sess["started_at"])
    elapsed = (datetime.now(timezone.utc) - started_at).total_seconds()
    time_limit = sess["time_limit_secs"]
    was_timed_out = elapsed > (time_limit + 5)

    if elapsed > (time_limit + 30):
        supabase.table("exam_sessions").update({"is_expired": True}).eq("id", session_id).execute()
        log_audit(profile["id"], "late_submission", {}, ip)
        raise HTTPException(status_code=410, detail="Submission rejected: time exceeded")

    supabase.table("exam_sessions").update({
        "submitted_at": datetime.now(timezone.utc).isoformat()
    }).eq("id", session_id).execute()

    background_tasks.add_task(
        _create_attempt,
        user_id=profile["id"],
        session_id=session_id,
        question_id=sess["question_id"],
        section=sess["section"],
        task=sess["task"],
        answer=body.answer,
        time_taken=int(elapsed),
        was_timed_out=was_timed_out,
        tab_penalties=sess.get("tab_switches", 0)
    )

    return {"status": "submitted", "time_taken": int(elapsed), "was_timed_out": was_timed_out}


@router.get("/{session_id}/time")
async def get_time_remaining(session_id: str, profile=Depends(get_current_user)):
    supabase = get_supabase()
    sess = supabase.table("exam_sessions")\
        .select("started_at,time_limit_secs,is_expired,submitted_at")\
        .eq("id", session_id)\
        .eq("user_id", profile.id)\
        .single()\
        .execute()

    if not sess.data:
        raise HTTPException(status_code=404, detail="Session not found")

    s = sess.data
    if s["is_expired"] or s["submitted_at"]:
        return {"time_remaining": 0, "expired": True}

    elapsed = (datetime.now(timezone.utc) - _parse_dt(s["started_at"])).total_seconds()
    remaining = max(0, s["time_limit_secs"] - int(elapsed))

    if remaining <= 0:
        supabase.table("exam_sessions").update({"is_expired": True}).eq("id", session_id).execute()
        return {"time_remaining": 0, "expired": True}

    return {"time_remaining": remaining, "expired": False}


@router.post("/tab-switch")
async def report_tab_switch(session_id: str, count: int, request: Request, profile=Depends(get_current_user)):
    supabase = get_supabase()
    ip = get_client_ip(request)
    count = max(0, min(int(count), 50))
    supabase.table("exam_sessions").update({"tab_switches": count}).eq("id", session_id).eq("user_id", profile.id).execute()
    log_audit(profile.id, "tab_switch", {"count": count}, ip)
    return {"recorded": True}


def _parse_dt(s: str):
    from dateutil import parser
    dt = parser.parse(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _create_attempt(user_id, session_id, question_id, section, task, answer, time_taken, was_timed_out, tab_penalties):
    supabase = get_supabase()
    supabase.table("attempts").insert({
        "session_id": session_id,
        "user_id": user_id,
        "question_id": question_id,
        "section": section,
        "task": task,
        "answer": answer,
        "time_taken": time_taken,
        "was_timed_out": was_timed_out,
        "tab_penalties": tab_penalties,
    }).execute()