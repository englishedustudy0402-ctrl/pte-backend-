import os, json, re, logging
import httpx
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
from middleware.security import require_authenticated
from services.groq_rotation import chat_json as groq_chat_json, available as groq_available, label as groq_label

load_dotenv()

router = APIRouter()
logger = logging.getLogger(__name__)

ANTHROPIC_API = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

SST_SYSTEM = """You are a certified PTE Academic "Summarize Spoken Text" examiner.
Score the student's summary against the official criteria, each out of a fixed maximum:

- content: 0-2 scale  (how accurately and completely it captures the transcript's main points and key supporting ideas)
- form: 0-1           (50-70 words; count words: rule of thumb 1 word = 5 characters including spaces)
- grammar: 0-2        (range and correctness)
- vocabulary: 0-2     (range and precision; formal academic register)
- spelling: 0-1       (no misspellings)

Be demanding but fair. Judge only what the student actually wrote.

Return ONLY valid JSON with exactly this shape, no markdown:
{"criteria": {
  "content":    {"score": 0, "comment": "plain string quoting the student's words and the transcript's facts"},
  "form":       {"score": 0, "comment": "plain string stating the word count and whether it met 50-70"},
  "grammar":    {"score": 0, "comment": "plain string quoting grammar errors or praising range"},
  "vocabulary": {"score": 0, "comment": "plain string noting register and word choice"},
  "spelling":   {"score": 0, "comment": "plain string listing any misspellings"}
 },
 "total": 0,
 "band": "string (e.g. 'Band 8: Distinction', 'Band 6: Competent')"}
Keep comments under 40 words each and never mention official PTE score bands you were not asked for."""


class SstScoreRequest(BaseModel):
    model: Optional[str] = ANTHROPIC_MODEL
    max_tokens: Optional[int] = 1000
    transcript: str = ""
    summary: str = ""
    difficulty: str = "medium"
    tier: int = 2


def _clean_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    raw = match.group() if match else text
    return json.loads(raw)


def _normalize(doc: dict, difficulty: str, tier: int, engine_label: str) -> dict:
    crit_in = doc.get("criteria") if isinstance(doc.get("criteria"), dict) else {}
    order = ("content", "form", "grammar", "vocabulary", "spelling")
    caps = {"content": 2, "form": 1, "grammar": 2, "vocabulary": 2, "spelling": 1}
    criteria = {}
    total = 0
    for key in order:
        cell = crit_in.get(key) if isinstance(crit_in.get(key), dict) else {}
        try:
            score = int(round(float(cell.get("score", 0))))
        except (TypeError, ValueError):
            score = 0
        score = max(0, min(caps[key], score))
        comment = str(cell.get("comment") or "").strip()
        criteria[key] = {"score": score, "comment": comment, "max": caps[key]}
        total += score
    total = max(0, min(8, total))
    band = str(doc.get("band") or _default_band(total))
    return {
        "total": total,
        "band": band,
        "criteria": criteria,
        "ai_engine": engine_label,
        "score_source": "anthropic",
        "difficulty": difficulty,
        "tier": tier,
    }


def _default_band(total: int) -> str:
    if total >= 7:
        return "Band 8: Distinction"
    if total >= 5:
        return "Band 6: Competent"
    if total >= 3:
        return "Band 4: Developing"
    return "Band 2: Needs attention"


def _sst_prompt(transcript: str, summary: str, difficulty: str, tier: str) -> str:
    tier_note = (
        "This is a PRACTICE item. Mark generously within the fixed maxima."
        if str(tier) == "1"
        else "This is an EXAM-LEVEL item. Mark to full official standard."
    )
    return (
        f"Audio transcript:\n{transcript}\n\n"
        f"Student summary ({len(summary.strip().split())} words):\n{summary}\n\n"
        f"Item difficulty: {difficulty}. {tier_note}"
    )


@router.post("/messages")
async def proxy_sst_score(body: SstScoreRequest, _=Depends(require_authenticated)):
    """Score the SST summary with Anthropic first, then Groq (rotating keys)
    when no Anthropic key is configured. Returns 503 only if neither is set so
    the client can fall back to its manual rubric guide."""
    answer = _sst_prompt(body.transcript, body.summary, body.difficulty, str(body.tier))
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        try:
            payload = {
                "model": ANTHROPIC_MODEL,
                "max_tokens": max(1, min(body.max_tokens or 1000, 4096)),
                "system": SST_SYSTEM,
                "messages": [{"role": "user", "content": answer}],
            }
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            async with httpx.AsyncClient(timeout=90.0) as client:
                resp = await client.post(
                    f"{ANTHROPIC_API}/messages", json=payload, headers=headers
                )
                resp.raise_for_status()
                data = resp.json()
            text = data["content"][0]["text"]
            return _normalize(_clean_json(text), body.difficulty, body.tier, f"Anthropic ({ANTHROPIC_MODEL})")
        except Exception as e:
            logger.error(f"Anthropic call failed: {e}")

    if groq_available():
        try:
            doc = groq_chat_json(SST_SYSTEM, answer)
            return _normalize(doc, body.difficulty, body.tier, groq_label())
        except Exception as e:
            logger.error("Groq SST scoring failed: %s", e)
            raise HTTPException(status_code=502, detail="AI scoring unavailable")

    raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY is not configured")