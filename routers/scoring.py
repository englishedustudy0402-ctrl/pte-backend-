import os, json, re, hashlib, logging
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from pydantic import BaseModel
from typing import Optional, List
from middleware.security import get_supabase, require_authenticated, optional_profile
from dotenv import load_dotenv
from supabase import create_client
from services.groq_rotation import chat_json as groq_chat_json, available as groq_available, label as groq_label, transcribe as groq_transcribe, whisper_label as groq_whisper_label
from services import audio_analysis
from services import equating as equating_svc
from services import phoneme as phoneme_svc

load_dotenv()

router = APIRouter()
logger = logging.getLogger(__name__)

DEFLATION = 0.72
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
def llm_engine_label() -> str:
    if groq_available():
        return groq_label()
    return "offline"

def chat_json(system: str, user: str) -> tuple[dict, str]:
    """Unified LLM call via Groq (rotating keys) — the only AI provider used.
    Returns (result_dict, engine_label) where engine_label names the model that answered."""
    if not groq_available():
        raise RuntimeError("No AI provider configured (set GROQ_API_KEYS)")
    return groq_chat_json(system, user), groq_label()

def deflate(score: int) -> int:
    return max(10, round(score * DEFLATION))

def pte_estimate(raw: int) -> str:
    return f"{round(raw * 1.05)}-{round(raw * 1.18)}"

# --- Writing ---
class WritingRequest(BaseModel):
    attempt_id: Optional[int] = None
    task: Optional[str] = "essay"
    prompt: Optional[str] = "General Practice"
    content: str
    word_min: Optional[int] = 200
    word_max: Optional[int] = 300

@router.post("/writing")
async def score_writing(body: WritingRequest, _=Depends(require_authenticated)):
    word_count = len(body.content.strip().split())
    prompt = f"""You are a PTE Academic examiner.
Task: {body.task}. Prompt: {body.prompt}
Submission: {body.content}
Word count: {word_count} ({body.word_min}-{body.word_max} required)
Return ONLY JSON: {{"score": 0-90, "grammar": 0-90, "vocabulary": 0-90, "coherence": 0-90, "content": 0-90, "errors": ["string"], "tip": "", "model": ""}}"""
    try:
        result = dict(chat_json("", prompt)[0])
        raw = result.get("score", 0)
        result["score"] = deflate(raw)
        result["pte_estimate"] = pte_estimate(raw)
        errors = result.get("errors", [])
        result["errors"] = [str(e) if not isinstance(e, str) else e for e in errors]
        return result
    except Exception as e:
        logger.error("Writing scoring failed: %s", e)
        raise HTTPException(status_code=500, detail="Scoring failed")


# --- MCQ ---
class McqRequest(BaseModel):
    attempt_id: Optional[int] = None
    user_selections: List[int]
    correct_answers: List[int]
    is_multi: Optional[bool] = False

@router.post("/reading/mcq")
async def score_mcq(body: McqRequest, _=Depends(require_authenticated)):
    correct = set(body.correct_answers)
    selected = set(body.user_selections)
    hits = len(correct & selected)
    total = len(correct)
    score = round((hits / total) * 90) if total > 0 else 0
    errors = []
    if selected != correct:
        errors.append(f"You selected {len(selected)} answer(s), {hits} were correct.")
    return {
        "score": deflate(score),
        "pte_estimate": pte_estimate(score),
        "hits": hits,
        "total": total,
        "errors": errors,
        "tip": "Read all options carefully before selecting." if hits < total else "Perfect selection!"
    }


# --- Reorder ---
class ReorderRequest(BaseModel):
    attempt_id: Optional[int] = None
    user_order: List[int]
    correct_order: List[int]

@router.post("/reading/reorder")
async def score_reorder(body: ReorderRequest, _=Depends(require_authenticated)):
    correct = 0
    for i, val in enumerate(body.user_order):
        if i < len(body.correct_order) and val == body.correct_order[i]:
            correct += 1
    total = len(body.correct_order)
    score = round((correct / total) * 90) if total > 0 else 0
    return {
        "score": deflate(score),
        "pte_estimate": pte_estimate(score),
        "hits": correct,
        "total": total,
        "errors": [] if correct == total else [f"{correct}/{total} sentences in correct position."],
        "tip": "Focus on logical flow and connecting words."
    }


# --- Fill Blanks ---
class FillBlanksRequest(BaseModel):
    attempt_id: Optional[int] = None
    user_answers: dict
    correct_answers: dict

@router.post("/reading/fill-blanks")
async def score_fill_blanks(body: FillBlanksRequest, _=Depends(require_authenticated)):
    correct = 0
    total = len(body.correct_answers)
    errors = []
    for key, ans in body.correct_answers.items():
        user = body.user_answers.get(key, "")
        if user.strip().lower() == ans.strip().lower():
            correct += 1
        else:
            errors.append(f"Blank {key}: you chose '{user}', correct is '{ans}'")
    score = round((correct / total) * 90) if total > 0 else 0
    return {
        "score": deflate(score),
        "pte_estimate": pte_estimate(score),
        "hits": correct,
        "total": total,
        "errors": errors,
        "tip": "Pay attention to grammar and context clues."
    }


# --- Dictation ---
class DictationRequest(BaseModel):
    attempt_id: Optional[int] = None
    correct_text: str
    user_answer: str

@router.post("/listening/dictation")
async def score_dictation(body: DictationRequest, _=Depends(require_authenticated)):
    correct_words = body.correct_text.lower().split()
    user_words = body.user_answer.lower().split()
    hits = sum(1 for w in user_words if w in correct_words)
    total = len(correct_words)
    score = round((hits / total) * 90) if total > 0 else 0
    errors = []
    if hits < total:
        errors.append(f"{hits}/{total} words matched.")
    return {
        "score": deflate(score),
        "pte_estimate": pte_estimate(score),
        "hits": hits,
        "total": total,
        "correct_text": body.correct_text,
        "errors": errors,
        "tip": "Listen for function words and endings carefully."
    }


# --- Speaking ---
class SpeakingAnalyzeRequest(BaseModel):
    attempt_id: Optional[int] = None
    task: str = "read_aloud"
    reference_text: str = ""
    transcript: str = ""
    duration_s: Optional[float] = None
    question_text: Optional[str] = ""

SPEAKING_SYSTEM = """You are a senior PTE Academic speaking examiner trained on the official Pearson scoring rubrics.
Score strictly like the real test: Read Aloud/CONTENT counts word-level errors (replacements, omissions, insertions); Repeat Sentence CONTENT is the % of words in correct sequence; Describe Image and Re-tell Lecture CONTENT judges how completely and accurately the main points were covered; Answer Short Question is correct/incorrect.
Pronunciation (0-5) and Oral Fluency (0-5) follow the official band descriptors.
Return ONLY JSON, no markdown, with this shape:
{"transcript":"", "content":0, "fluency":0, "pronunciation":0, "errors":["plain string errors, be specific, quote the missing/mispronounced word and say its correct pronunciation"],
"tip":"one actionable trick", "model":"an ideal full response for this item"}
All scores 0-90. Be demanding but fair. The transcript may contain recognition noise — judge only what you can hear."""

@router.post("/speaking/analyze")
async def score_speaking_text(body: SpeakingAnalyzeRequest, _=Depends(optional_profile)):
    try:
        result, engine = chat_json(
            SPEAKING_SYSTEM,
            f"Task: {body.task}\nReference: {body.reference_text}\nExtra context: {body.question_text}\nStudent transcript: {body.transcript}\nSpeaking duration in seconds: {body.duration_s}",
        )
        raw = result.get("content", 0) or 0
        result["content"] = deflate(float(raw))
        result["fluency"] = deflate(float(result.get("fluency", 0) or 0))
        result["pronunciation"] = deflate(float(result.get("pronunciation", 0) or 0))
        overall = round((result["content"] + result["fluency"] + result["pronunciation"]) / 3)
        result["score"] = overall
        result["pte_estimate"] = pte_estimate(overall)
        result["errors"] = [str(e) for e in result.get("errors", [])][:6]
        result["ai_engine"] = engine
        return result
    except Exception as e:
        logger.error(f"Speaking analyze failed: {e}")
        raise HTTPException(status_code=503, detail="AI feedback unavailable")


# --- PTE Academic Listening examiner (AI commentary over deterministic score) ---
class ExaminerRequest(BaseModel):
    task: str = "dictation"
    question: str = ""
    user_answer: str = ""
    reference: str = ""
    max_score: int = 10
    # Official-mirror deterministic result computed by the client's scoring engine
    # (same logic the rule-based fallback uses). Numbers here are authoritative;
    # the AI is only allowed to write prose around them.
    deterministic: Optional[dict] = None

COMMENTARY_SYSTEM = """You are a certified PTE Academic listening examiner giving coaching feedback on ONE student response.

The SCORES have already been computed by an official-style scoring engine and are given to you. Never invent, compute or change any score — your job is only to write short, specific, encouraging commentary that is consistent with those scores.

Return ONLY valid JSON, no markdown, with exactly these keys:
{"tip": "<one specific, actionable study trick, tailored to the task type and to what the student actually got wrong>",
 "encouragement": "<a short, sincere, non-generic encouragement that reflects the score>",
 "criteria": {"content": {"comment": "<string>"},
              "form": {"comment": "<string>"},
              "grammar": {"comment": "<string>"},
              "vocabulary": {"comment": "<string>"},
              "spelling": {"comment": "<string>"}}}

Rules:
1. Every criterion comment must quote the student's actual words/answers from the "Student response" and compare them with the correct answer from "Audio transcript / reference".
2. Comments must be consistent with the given criterion scores: a high-scoring criterion gets positive language, a low-scoring one gets a concrete correction.
3. If the student response is empty or clearly not attempted, say so bluntly but kindly and give a fresh-start tip.
4. Never mention scores, points, percentages or bands in the comments — just the content-level feedback.
5. The tip must be tailored to the task type (SST, MCQ multi, Fill in the Blanks, HCS, MCQ single, Missing Word, Highlight Incorrect Words, Write from Dictation)."""

EXAM_CRITERIA = ('content', 'form', 'grammar', 'vocabulary', 'spelling')


class CoachRequest(BaseModel):
    section: str = "listening"          # reading | writing | speaking | listening
    task: str = ""
    question: str = ""                  # prompt / passage / question text
    reference: str = ""                 # correct answer / transcript / model answer
    student_response: str = ""          # what the student actually answered
    note: str = ""                      # deterministic score summary (official-mirror)


_COACH_CACHE: dict = {}
_COACH_CACHE_MAX = 200


COACH_SYSTEM = """You are a certified PTE Academic examiner giving LIVE coaching feedback on ONE student's actual response for the given task.

The official-style scores are computed by the app's deterministic engine and are given to you only as context. NEVER invent, compute or change any score — your job is to analyse the student's ACTUAL response.

Rules:
1. Always compare the student's real words or selections with the reference / correct answer, and quote them directly (e.g. "you selected 'B', which was correct" or "you wrote 'sayed', the correct form is 'said'").
2. Be concrete and specific to THIS response — never generic study advice. Do NOT produce pre-written tips.
3. If the response is empty or clearly not attempted, say so kindly and give a practical fresh-start tip.
4. Return ONLY valid JSON with exactly these keys:
{"feedback": "<2-4 sentence paragraph analysing the actual response>",
 "tip": "<one specific, actionable improvement for this exact submission>",
 "encouragement": "<one short, sincere, non-generic line>"}
5. No markdown, no scores, no percentages, no bands in the text."""


@router.post("/generate/coach")
async def generate_coach(body: CoachRequest, _=Depends(optional_profile)):
    """Live AI coaching commentary for ANY task (reading/writing/speaking/listening).
    Deterministic scores are context-only; the AI comments on the real response."""
    if not (body.student_response or "").strip() and not (body.reference or "").strip():
        return {
            "feedback": "No response was recorded for this answer.",
            "tip": "Read the question carefully, then attempt it before asking for feedback.",
            "encouragement": "Try the next one — practice is how exam skills grow.",
            "ai_engine": "offline",
            "score_source": "official-mirror (deterministic)",
        }
    cache_key = hashlib.sha256(
        f"{body.section}|{body.task}|{body.question}|{body.reference}|{body.student_response}|{body.note}".encode()
    ).hexdigest()
    cached = _COACH_CACHE.get(cache_key)
    if cached:
        return {**cached, "cached": True}
    user = (
        f"Section: {body.section}\n"
        f"Task: {body.task}\n"
        f"Question / prompt: {body.question}\n"
        f"Reference / correct answer: {body.reference}\n"
        f"Student's actual response: {body.student_response}\n"
        f"Official result (context only): {body.note}"
    )
    try:
        raw, engine = chat_json(COACH_SYSTEM, user)
        doc = {
            "feedback": str(raw.get("feedback") or ""),
            "tip": str(raw.get("tip") or ""),
            "encouragement": str(raw.get("encouragement") or ""),
            "ai_engine": engine,
            "score_source": "official-mirror (deterministic)",
        }
        _COACH_CACHE[cache_key] = doc
        if len(_COACH_CACHE) > _COACH_CACHE_MAX:
            oldest = next(iter(_COACH_CACHE))
            _COACH_CACHE.pop(oldest, None)
        return doc
    except Exception as e:
        logger.error(f"Live coach failed: {e}")
        raise HTTPException(status_code=503, detail="AI coaching unavailable")


def _examiner_int(value, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _merge_examiner(base: dict, llm: dict, max_score: int, engine: str) -> dict:
    """Deterministic numbers come from `base`; prose comes from the AI."""
    if not isinstance(base, dict):
        base = {}
    max_score = max(1, _examiner_int(max_score, 10))
    score = max(0, min(max_score, _examiner_int(base.get('score'), 0)))
    pct = _examiner_int(base.get('percentage'), round(score / max_score * 100) if max_score else 0)
    pct = max(0, min(100, pct))
    band = str(base.get('band') or 'Intermediate')
    crit = base.get('criteria') if isinstance(base.get('criteria'), dict) else {}
    if not isinstance(llm, dict):
        llm = {}
    llm_crit = llm.get('criteria') if isinstance(llm.get('criteria'), dict) else {}
    criteria = {}
    for k in EXAM_CRITERIA:
        cell = crit.get(k) if isinstance(crit.get(k), dict) else {}
        cscore = max(0, min(max_score, _examiner_int(cell.get('score'), 0)))
        cell2 = llm_crit.get(k) if isinstance(llm_crit.get(k), dict) else {}
        comment = str(cell2.get('comment') or cell.get('comment') or '')
        criteria[k] = {"score": cscore, "max": max_score, "comment": comment}
    return {
        "score": score,
        "max_score": max_score,
        "band": band,
        "percentage": pct,
        "criteria": criteria,
        "tip": str(llm.get('tip') or base.get('tip') or ''),
        "model_answer": str(base.get('model_answer') or llm.get('model_answer') or ''),
        "encouragement": str(llm.get('encouragement') or base.get('encouragement') or ''),
        "ai_engine": engine,
        "score_source": "official-mirror (deterministic)",
    }

@router.post("/generate/examiner")
async def generate_examiner(body: ExaminerRequest, _=Depends(require_authenticated)):
    """AI coach commentary layered over the deterministic official-mirror score."""
    base = body.deterministic if isinstance(body.deterministic, dict) else None
    if not base:
        raise HTTPException(status_code=400, detail="deterministic scoring payload is required")
    max_score = body.max_score or base.get("max_score") or 10
    prompt = (
        f"PTE Listening task: {body.task}\n"
        f"Question: {body.question}\n"
        f"Audio transcript / reference: {body.reference}\n"
        f"Max possible score: {max_score}\n"
        f"Student response: {body.user_answer}\n\n"
        f"OFFICIAL-MIRROR DETERMINISTIC SCORES (adjust your commentary to these, never change them):\n"
        f"{json.dumps(base, default=str)}"
    )
    try:
        raw, engine = chat_json(COMMENTARY_SYSTEM, prompt)
        return _merge_examiner(base, raw, max_score, engine)
    except Exception as e:
        logger.error(f"Examiner generation failed: {e}")
        raise HTTPException(status_code=503, detail=f"AI feedback unavailable: {e}")


@router.post("/speaking/upload")
async def score_speaking(
    attempt_id: Optional[str] = None,
    task: str = "read_aloud",
    reference_text: str = "",
    difficulty: str = "",
    audio: UploadFile = File(...),
    profile=Depends(optional_profile),
):
    if attempt_id and not re.match(r"^[A-Za-z0-9_-]{1,64}$", attempt_id):
        raise HTTPException(status_code=400, detail="Invalid attempt id")
    # Browsers append codec params to the MIME (e.g. "audio/webm;codecs=opus"),
    # so compare the bare media type. If it's a recognised audio MIME we accept
    # it; Whisper/ffmpeg handle the actual (re-)encoding.
    raw_ct = (audio.content_type or "").split(";")[0].strip().lower()
    if raw_ct and not raw_ct.startswith("audio/"):
        raise HTTPException(status_code=415, detail="Unsupported audio format")
    audio_bytes = await audio.read(MAX_UPLOAD_BYTES + 1)
    if len(audio_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Audio too large (max 25 MB)")
    audio_mime = (audio.content_type or "audio/webm").split(";")[0].strip().lower() or "audio/webm"
    user_id = (profile or {}).get("id")
    if attempt_id:
        if attempt_id == '0':
            attempt_id = None
        elif user_id:
            owner = (
                get_supabase().table("attempts")
                .select("id").eq("id", attempt_id).eq("user_id", user_id)
                .maybe_single().execute()
            )
            if not (owner and owner.data):
                raise HTTPException(status_code=403, detail="Not your attempt")

    try:
        # Persist the raw recording to the private recordings bucket:
        # recordings/{user_id}/{attempt_id}/audio.webm (server-authoritative
        # copy; signed URLs are issued by /recordings/*). Guests are skipped.
        try:
            if user_id:
                storage = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_ROLE_KEY"))
                key = f"{user_id}/{attempt_id or 'draft'}/audio.webm"
                storage.storage.from_("recordings").upload(key, audio_bytes, {
                    "content-type": audio.content_type or "audio/webm",
                    "upsert": "true",
                })
                if attempt_id:
                    get_supabase().table("attempts").update({"audio_path": key}).eq("id", attempt_id).eq("user_id", user_id).execute()
        except Exception as e:
            logger.warning("recording persistence failed: %s", e)

        transcript: str | None = None
        result: dict | None = None
        engine_label = ""
        audio_feats: dict = {}

        # Tier 1 — measure the SOUND, not just the words. Decode the waveform and
        # extract real acoustic features (VAD silences, speech rate, pitch) that
        # become the authority for fluency/pronunciation rather than the LLM's
        # guess from the transcript. This is the Pearson-style "score the audio".
        try:
            audio_feats = audio_analysis.analyze(audio_bytes, audio_mime, reference_text, transcript="")
        except Exception as e:
            logger.warning("audio feature extraction failed: %s", e)
            audio_feats = {}

        # Groq-only path: Whisper transcribes the voice, then a Groq chat model
        # scores it. If Groq is unavailable the request fails — no hidden fallback.
        if not groq_available():
            raise HTTPException(status_code=503, detail="AI feedback unavailable")
        try:
            transcript = groq_transcribe(audio_bytes, audio_mime)
            # attach the real transcript so speech-rate can use the true word count
            if audio_feats:
                audio_feats["words"] = len([w for w in transcript.split() if w.strip()])
            system = (
                "You are a senior PTE Academic speaking examiner. Score the "
                "student's TRANSCRIPT strictly like the real test for the given "
                "task (word-level errors for Read Aloud, correct sequence for "
                "Repeat Sentence, key-point coverage for Describe Image and "
                "Re-tell Lecture, correct/incorrect for Answer Short Question). "
                "Return ONLY JSON, no markdown, with exactly this shape: "
                '{"transcript": "", "content": 0, "fluency": 0, "pronunciation": 0, '
                '"errors": ["plain string errors, quote missing/mispronounced words and give the correct pronunciation"], '
                '"tip": "one actionable trick", "model": "an ideal full response for this item"}. '
                "All scores 0-90. Be demanding but fair; the transcript may contain "
                "recognition noise — judge only what you can hear."
            )
            user = (
                f"Task: {task}\nReference: {reference_text}\n"
                f"Student transcript: {transcript}"
            )
            scored = groq_chat_json(system, user)
            scored = dict(scored or {})
            scored.setdefault("transcript", transcript)
            result = scored
            engine_label = f"{groq_whisper_label()} + {groq_label()}"
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Groq speaking upload path failed: %s", e)
            raise HTTPException(status_code=503, detail="AI feedback unavailable")

        # Tier 1 blending — intersect the LLM scores with the WAVEFORM facts.
        # The LLM sees words; the audio tells the truth about delivery. When the
        # waveform says fluency/pronunciation are weak, cap the scored values so
        # a fluent-looking transcript cannot mask a disfluent, flat recording.
        if audio_feats.get("available"):
            fl = audio_analysis.fluency_from_features(audio_feats)
            pr = audio_analysis.pronunciation_from_features(audio_feats)
            fl_rate = fl.get("fluency01", 0.0)
            pr_rate = pr.get("pron01", 0.0)
            cur_fl = deflate(float(result.get("fluency") or 0))
            cur_pr = deflate(float(result.get("pronunciation") or 0))
            # waveform-capped: audio bad -> score can't stay high
            capped_fl = min(cur_fl, round(max(20.0, fl_rate * 88)))
            capped_pr = min(cur_pr, round(max(20.0, pr_rate * 88)))
            result["fluency"] = capped_fl
            result["pronunciation"] = capped_pr
            if fl.get("reasons"):
                result.setdefault("errors", []).extend(fl["reasons"][:3])
            if pr.get("reasons"):
                result.setdefault("errors", []).extend(pr["reasons"][:3])
            # expose the audio backbone to the client
            result["audio_features"] = audio_feats
            result["audio_fluency"] = fl
            result["audio_pronunciation"] = pr

            # Tier 1b — phoneme-level pronunciation. Decode the PCM once and
            # score each transcript word from its own acoustic window (voicing,
            # envelope smoothness, pitch continuity) when we have voiced audio
            # and words to place. Uses forced alignment when available (Tier 4c),
            # otherwise the acoustic-fused local path.
            try:
                pcm_sig, pcm_rate = audio_analysis.decode_to_pcm(audio_bytes, audio_mime)
                voiced_s = float(audio_feats.get("voiced_s") or 0.0)
                words = [w for w in transcript.split() if w.strip()]
                if pcm_sig is not None and voiced_s > 0 and len(words) > 0:
                    step = voiced_s / len(words)
                    timestamps = [
                        {"word": words[i], "start": round(i * step, 3),
                         "end": round((i + 1) * step, 3)}
                        for i in range(len(words))
                    ]
                    ph = phoneme_svc.phoneme_pronunciation(pcm_sig, pcm_rate, timestamps)
                    if ph["pron90"] > 10:
                        # authoritative pron: phoneme estimate overrides the
                        # coarse pitch-only heuristic when it has real words.
                        result["pronunciation"] = ph["pron90"]
                        result["phoneme_pronunciation"] = ph
            except Exception as e:
                logger.warning("phoneme pronunciation failed: %s", e)

        raw = float(result.get("score") or 0)
        if raw <= 0:
            parts = [
                float(result.get("content") or 0),
                float(result.get("fluency") or 0),
                float(result.get("pronunciation") or 0),
            ]
            raw = round(sum(parts) / len(parts)) if parts else 0
        result["score"] = deflate(raw)
        result["pte_estimate"] = pte_estimate(raw)
        result["ai_engine"] = engine_label
        errors = result.get("errors", [])
        result["errors"] = [str(e) if not isinstance(e, str) else e for e in errors]
        if not result.get("transcript") and transcript:
            result["transcript"] = transcript

        # Tier 3c — equate the capped traits onto the fixed 10..90 ruler so the
        # reported score means the same across difficulties (hard item done well
        # is worth more than the easy one). Exposed as `equated`.
        try:
            eq = equating_svc.equate_traits(
                {"content": result.get("content"), "fluency": result.get("fluency"),
                 "pronunciation": result.get("pronunciation")},
                difficulty=difficulty or None,
                n_items=1,
            )
            eq["raweq"] = result["score"]
            result["equated"] = eq
        except Exception as e:
            logger.warning("equating failed: %s", e)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error("speaking upload scoring failed: %s", e)
        raise HTTPException(status_code=500, detail="Scoring failed")