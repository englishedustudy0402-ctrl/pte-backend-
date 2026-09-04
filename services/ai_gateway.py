"""
AI feedback gateway (STEP 10). Thin, schema-validated wrapper around the model
provider configured via environment keys. Deterministic scores are ALWAYS the
authority; the AI writes prose commentary only. Falls back gracefully when no
provider is reachable (caller decides).

Provider: Groq only (rotating keys). No Gemini/Grok/Ollama.
"""
import json
import logging
import re

from services.groq_rotation import chat_json as groq_chat_json, available as groq_available, label as groq_label

logger = logging.getLogger(__name__)

COMMENTARY_SCHEMA = {"feedback", "tip", "encouragement"}

COMMENTARY_SYSTEM = """You are a certified PTE Academic examiner giving feedback on ONE student response.
The official deterministic scores are supplied as context. NEVER invent or change scores.
Return ONLY valid JSON with exactly these keys:
{"feedback":"<2-4 sentences analysing the actual response, quoting the student's words/options and comparing to the reference>",
 "tip":"<one specific, actionable improvement for THIS exact submission>",
 "encouragement":"<one short, non-generic line that reflects the score>"}
No markdown, no scores, no percentages, no bands in the prose."""


def _extract_json(text):
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        return json.loads(match.group() if match else text)
    except (ValueError, TypeError):
        return None


def _validated(result):
    if not isinstance(result, dict):
        return None
    missing = COMMENTARY_SCHEMA - set(result.keys())
    if missing:
        logger.warning("AI response missing keys %s: %s", missing, str(result)[:200])
        return None
    return {
        "feedback": str(result.get("feedback") or ""),
        "tip": str(result.get("tip") or ""),
        "encouragement": str(result.get("encouragement") or ""),
    }


def commentary(task, question_text, reference, student_response, note=""):
    """Generate examiner commentary. Returns (doc, engine) where doc is a dict
    or None when no provider produced a schema-valid response."""
    prompt = (
        f"Task: {task}\n"
        f"Question / prompt: {question_text}\n"
        f"Reference / correct answer: {reference}\n"
        f"Student response: {student_response}\n"
        f"Official result (context only): {note}"
    )

    if groq_available():
        try:
            doc = _validated(groq_chat_json(COMMENTARY_SYSTEM, prompt))
            if doc:
                return doc, groq_label()
        except Exception as e:
            logger.warning("Groq commentary unavailable: %s", e)

    return None, "offline"