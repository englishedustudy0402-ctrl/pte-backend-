"""
Canonical deterministic-first scoring engine for the production pipeline.

Every task is scored on a 0..90 PTE-style scale. Deterministic formulas run
first (exact/word-level/keyword coverage); AI commentary is layered on later by
routers through ai_gateway. persist() writes the canonical record to
public.scores + public.score_components, updates the attempt's `score`/`status`
and rolls up public.user_statistics.
"""
import json
import logging
from difflib import SequenceMatcher
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

SCORING_ENGINE = "deterministic"
SCORING_ENGINE_VERSION = "1.0.0"
SCALE = 90


def _frac(value, scale=100):
    try:
        return max(0.0, min(1.0, float(value) / scale))
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def _score(fraction):
    return round(max(0, min(SCALE, round(fraction * SCALE))))


def _tokens(text):
    return (text or "").lower().split()


def _ratio(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def _keyword_coverage(keywords, answer):
    ans = set(_tokens(answer))
    if not keywords:
        return None
    total = len(keywords)
    if total == 0:
        return None
    hits = sum(1 for k in keywords if k.lower() in ans)
    return hits / total


def _parse_structured(answer, answer_text):
    """Unified answer -> python value. Accepts JSON-encoded lists/dicts (clients
    send selections as JSON strings) or plain text."""
    for candidate in (answer_text, answer):
        if not candidate:
            continue
        try:
            value = json.loads(candidate)
            if isinstance(value, (list, dict)):
                return value
        except (ValueError, TypeError):
            continue
    return answer_text or answer or ""


def _options_list(content):
    opts = content.get("options")
    if isinstance(opts, str):
        return [o.strip() for o in opts.split(",") if o.strip()]
    if isinstance(opts, list):
        return [str(o).strip() for o in opts]
    if isinstance(opts, dict):
        return [str(o).strip() for o in opts.get("options") or []]
    return []


def _normalize_indices(raw):
    """Coerce correct-answer references ('1', 1, [1], '[0,2]', ['1']) to ints."""
    if raw is None:
        return []
    if isinstance(raw, list):
        out = []
        for x in raw:
            try:
                out.append(int(x))
            except (TypeError, ValueError):
                pass
        return out
    s = str(raw).strip()
    if not s:
        return []
    if s.isdigit():
        return [int(s)]
    if s.startswith("["):
        try:
            return _normalize_indices(json.loads(s))
        except (TypeError, ValueError):
            return []
    return []


def _pick_mcq(question, payload):
    content = question.get("content") or {}
    answer_data = question.get("answer_data") or {}
    raw = answer_data.get("correct") if answer_data.get("correct") is not None else content.get("correct")
    correct = _normalize_indices(raw)
    opts = _options_list(content)
    sel = payload
    if isinstance(sel, dict):
        sel = sel.get("selections") or sel.get("selected") or []
    items = sel if isinstance(sel, list) else [sel]
    selected = []
    for s in items:
        if isinstance(s, (int, float)):
            selected.append(int(s))
            continue
        sd = str(s).strip()
        if sd.isdigit():
            selected.append(int(sd))
            continue
        if opts:
            for i, o in enumerate(opts):
                if sd and (sd.lower() in o.lower() or o.lower() in sd.lower()):
                    selected.append(i)
                    break
    hits = len([c for c in selected if c in correct])
    total = len(correct)
    return _score(hits / total) if total else SCALE, hits, total


def _pick_reorder(question, payload):
    if isinstance(payload, dict):
        payload = payload.get("order") or payload.get("user_order") or []
    user = [int(s) for s in (payload or [])]
    try:
        reference = [int(s) for s in list((question.get("answer_data") or {}).get("correct_order") or
                                          (question.get("content") or {}).get("correct") or [])]
    except (TypeError, ValueError):
        reference = []
    if not reference:
        return None, 0, 0
    hits = sum(1 for i, v in enumerate(user) if i < len(reference) and v == reference[i])
    return _score(hits / len(reference)) if reference else SCALE, hits, len(reference)


def _pick_fill(question, payload):
    answer_data = question.get("answer_data") or {}
    content = question.get("content") or {}
    blanks = dict(answer_data.get("blanks") or content.get("blanks") or {})
    user = payload if isinstance(payload, dict) else {}
    if not blanks:
        return None, 0, 0
    hits = total = 0
    for key, info in blanks.items():
        correct = (info or {}).get("correct") or ""
        total += 1
        if str(user.get(key) or "").strip().lower() == str(correct).strip().lower():
            hits += 1
    return _score(hits / total), hits, total


def _word_match(question, answer):
    reference = (question.get("answer_data") or {}).get("reference_text") or (question.get("content") or {}).get("text") or ""
    ref = _tokens(reference)
    usr = _tokens(answer)
    frac = _ratio(ref, usr)
    return _score(frac), frac, len(ref), len(usr)


def _blank_compare(question, answer):
    reference = _tokens((question.get("answer_data") or {}).get("reference_text") or (question.get("content") or {}).get("text") or "")
    usr = _tokens(answer)
    return _score(_ratio(reference, usr))


def _selection_index(question, payload):
    """For tasks whose answer is a single option index (hcs, missing_word)."""
    answer_data = question.get("answer_data") or {}
    content = question.get("content") or {}
    raw = answer_data.get("correct") if answer_data.get("correct") is not None else content.get("correct")
    reference = _normalize_indices(raw)
    sel = payload
    if isinstance(payload, dict):
        sel = payload.get("selection") or payload.get("selected")
    if isinstance(sel, (list, tuple)) and len(sel) == 1:
        sel = sel[0]
    if not reference:
        return None, False
    try:
        chosen = int(sel)
    except (TypeError, ValueError):
        return None, False
    return chosen, chosen in reference


def _incorrect_words(question, payload):
    answer_data = question.get("answer_data") or {}
    content = question.get("content") or {}
    wrong = set(answer_data.get("wrong_indices") or content.get("wrongIndices") or [])
    sel = payload

    def _to_set(v):
        out = set()
        for item in (v or []):
            try:
                out.add(int(item))
            except (TypeError, ValueError):
                pass
        return out

    if isinstance(sel, dict):
        selected = _to_set(sel.get("indices") or sel.get("selections"))
    else:
        selected = _to_set(sel if isinstance(sel, (list, tuple)) else [sel])
    if not wrong:
        return None, 0, 0
    hits = len(wrong & selected)
    false_positives = len(selected - wrong)
    total = len(wrong)
    penalty = max(0, total - false_positives - hits)
    score = _score((total - penalty) / total)
    return score, hits, total


def grade(question, attempt, body) -> dict:
    """Deterministic official-mirror grade. Returns {score, components, meta}."""
    task = attempt["task"]
    answer_text = body.get("answer_text")
    answer = body.get("answer") or answer_text or ""
    payload = _parse_structured(answer, answer_text)
    content = question.get("content") or {}
    answer_data = question.get("answer_data") or {}

    components = []
    meta = {}

    def add(name, raw, frac, feedback=""):
        components.append({
            "component": name,
            "raw_score": raw if raw is None or isinstance(raw, int) else round(raw),
            "normalized_score": _score(frac) if frac is not None else None,
            "feedback": feedback,
        })

    answer_norm = " ".join(_tokens(answer))
    attempted = bool(answer_norm)

    if task in ("mcq", "mcq_multi"):
        score, hits, total = _pick_mcq(question, payload)
        meta = {"hits": hits, "total": total}
        if score is not None:
            add("accuracy", hits, hits / total if total else 0,
                f"{hits}/{total} selections correct." if hits < total else "Perfect selection.")
    elif task == "reorder":
        score, hits, total = _pick_reorder(question, payload)
        meta = {"hits": hits, "total": total}
        if score is not None:
            add("accuracy", hits, hits / total, f"{hits}/{total} sentences in the correct position.")
    elif task in ("fill_blanks", "rw_fill_blanks"):
        score, hits, total = _pick_fill(question, payload)
        meta = {"hits": hits, "total": total}
        if score is not None:
            add("accuracy", hits, hits / total, f"{hits}/{total} blanks correct.")
    elif task in ("hcs", "missing_word"):
        chosen, ok = _selection_index(question, payload)
        meta = {"selected": chosen, "correct": ok}
        score = _score(1.0) if ok else 0
        if not attempted and choice_missing(payload):
            score = 0
        add("accuracy", 1 if ok else 0, 1.0 if ok else 0.0, "Correct selection." if ok else "Incorrect selection.")
    elif task == "incorrect_words":
        score, hits, total = _incorrect_words(question, payload)
        meta = {"hits": hits, "total": total}
        if score is not None:
            add("accuracy", hits, hits / total, f"{hits}/{total} wrong words identified.")
    elif task in ("dictation", "read_aloud", "repeat_sentence"):
        score, frac, ref_n, usr_n = _word_match(question, answer)
        meta = {"reference_words": ref_n, "answer_words": usr_n, "word_accuracy": round(frac, 3)}
        add("content", score, frac, f"Word-level accuracy {usr_n}/{ref_n}." if ref_n else "")
    elif task in ("sst", "summarize", "describe_image", "retell_lecture",
                  "respond_to_situation", "summarize_group_discussion", "essay"):
        keywords = answer_data.get("keywords") or content.get("keywords") or []
        frac_kw = _keyword_coverage(keywords, answer)
        wc = len(_tokens(answer))
        if frac_kw is None:
            frac_kw = 0.0 if not attempted else 0.5
            feedback = "Open task: keyword baseline unavailable; scored on attempt completeness."
        else:
            feedback = f"Covered {frac_kw:.0%} of the key points."
        score = _score(frac_kw)
        meta = {"keywords": len(keywords), "word_count": wc, "coverage": round(frac_kw, 3)}
        add("content", score, frac_kw, feedback)
    elif task == "answer_short":
        corr = (answer_data.get("correct_answers") or answer_data.get("answer")
                or content.get("answer") or content.get("correct"))
        corrects = [str(corr)] if isinstance(corr, str) else [str(c) for c in (corr or [])]
        ok = any(_ratio(_tokens(a), _tokens(answer)) >= 0.85 for a in corrects)
        meta = {"correct": ok}
        score = _score(1.0) if ok else 0
        add("content", 1 if ok else 0, 1.0 if ok else 0.0, "Correct." if ok else "Incorrect.")
    else:
        score = _score(0.5) if attempted else 0
        meta = {"note": f"deterministic fallback for task={task}"}
        add("content", score, 0.5 if attempted else 0.0, "Deterministic baseline.")

    if not attempted:
        score = 0

    body_data = {
        "overall_score": score,
        "engine": SCORING_ENGINE,
        "version": SCORING_ENGINE_VERSION,
        "components": components,
        "meta": meta,
    }
    return body_data


def choice_missing(payload):
    return payload in (None, "", [], {})


def persist(supabase, attempt, question, grade_result, extra_text=""):
    """Write grade_result into scores/score_components and update the attempt +
    user_statistics. Returns the score row id."""
    score = int(grade_result["overall_score"] or 0)
    now = datetime.now(timezone.utc).isoformat()

    scores = supabase.table("scores").insert({
        "attempt_id": attempt["id"],
        "overall_score": score,
        "scoring_engine": grade_result["engine"],
        "scoring_engine_version": grade_result["version"],
        "confidence": 0.9,
        "feedback": extra_text or json.dumps(grade_result["meta"], default=str),
    }).execute()
    score_id = scores.data[0]["id"]

    components = grade_result.get("components") or []
    if components:
        rows = [
            {
                "score_id": score_id,
                "component": c["component"],
                "raw_score": c.get("raw_score"),
                "normalized_score": c.get("normalized_score"),
                "feedback": c.get("feedback", ""),
            }
            for c in components
        ]
        for chunk in (rows[i:i + 100] for i in range(0, len(rows), 100)):
            supabase.table("score_components").insert(chunk).execute()

    supabase.table("attempts").update({
        "score": score,
        "status": "submitted",
        "answer_text": attempt.get("answer_text"),
    }).eq("id", attempt["id"]).execute()

    _rollup_stats(supabase, attempt["user_id"], attempt["task"], score)
    return score_id


def _rollup_stats(supabase, user_id, task, latest_score):
    scores = supabase.table("attempts").select("score") \
        .eq("user_id", user_id).eq("task", task).eq("status", "submitted").execute()
    values = [float(s.get("score") or 0) for s in (scores.data or [])]
    if not values:
        return
    avg = round(sum(values) / len(values), 1)
    best = max(values)
    supabase.table("user_statistics").upsert({
        "user_id": user_id,
        "task_type": task,
        "attempts": len(values),
        "average_score": avg,
        "best_score": best,
        "recent_score": latest_score,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }, on_conflict="user_id,task_type").execute()