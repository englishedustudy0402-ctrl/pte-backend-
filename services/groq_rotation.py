"""
Groq key rotation (OpenAI-compatible). GROQ_API_KEYS holds a comma-separated
list of API keys. Every call picks the next key in round-robin order; if a key
hits a 429 / timeout it is cooled down for 30 s and the next key is tried
automatically. Clients are cached per key; never logged.
"""
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from typing import Generator

from openai import OpenAI

logger = logging.getLogger(__name__)

# ── defaults ────────────────────────────────────────────────────────────────

def _default_model() -> str:
    return os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

def _whisper_model() -> str:
    return os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo")

# ── key list + round-robin ──────────────────────────────────────────────────

def keys() -> list[str]:
    raw = os.getenv("GROQ_API_KEYS", "")
    return [k.strip() for k in raw.split(",") if k.strip()]

def available() -> bool:
    return len(keys()) > 0

_clients: dict[str, OpenAI] = {}
_rr_idx: int = 0                 # round-robin counter
_busy: dict[str, float] = {}     # key → cooldown-until timestamp
_COOLDOWN = 30.0                 # seconds to skip a rate-limited key

def _next_key() -> str:
    """Pick the next non-busy key (round-robin with cooldown skip)."""
    global _rr_idx
    ks = keys()
    if not ks:
        raise RuntimeError("GROQ_API_KEYS is not configured")
    now = time.time()
    for _ in range(len(ks)):
        k = ks[_rr_idx % len(ks)]
        _rr_idx += 1
        expiry = _busy.get(k, 0.0)
        if expiry <= now:
            return k
    # all keys busy → return the one closest to expiring (least penalty)
    return min(ks, key=lambda k: _busy.get(k, 0.0))

def _mark_busy(key: str):
    """Mark key as rate-limited so it is skipped for _COOLDOWN seconds."""
    _busy[key] = time.time() + _COOLDOWN
    logger.warning("key …%s marked busy for %.0fs", key[-4:], _COOLDOWN)

def client() -> OpenAI:
    """Return an OpenAI client on the next available key."""
    key = _next_key()
    c = _clients.get(key)
    if c is None:
        c = OpenAI(
            api_key=key,
            base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
            timeout=60.0,
            max_retries=0,  # we handle retries ourselves
        )
        _clients[key] = c
    return c

def client_for(key: str) -> OpenAI:
    """Return a client pinned to a specific key (used by transcribe fallback)."""
    c = _clients.get(key)
    if c is None:
        c = OpenAI(
            api_key=key,
            base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
            timeout=60.0,
            max_retries=0,
        )
        _clients[key] = c
    return c

# ── helpers ─────────────────────────────────────────────────────────────────

def _is_retryable(exc: Exception) -> bool:
    """True if the error is a 429 / rate-limit / timeout."""
    msg = str(exc).lower()
    code = getattr(exc, "status_code", None)
    if code == 429:
        return True
    if code in (408, 503, 504):
        return True
    if any(s in msg for s in ("429", "rate limit", "timeout", "timed out", "overloaded")):
        return True
    return False

def _default_max_tokens() -> int:
    return int(os.getenv("GROQ_MAX_TOKENS", "1024"))

# ── chat_json with failover ────────────────────────────────────────────────

def chat_json(system: str, user: str, *, max_tokens: int | None = None) -> dict:
    """One Groq chat call with automatic key failover on 429/timeout."""
    keys_list = keys()
    attempts = min(4, len(keys_list))
    last_err: Exception | None = None
    for i in range(attempts):
        k = _next_key()
        c = client_for(k)
        try:
            resp = c.chat.completions.create(
                model=_default_model(),
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=max_tokens or _default_max_tokens(),
            )
            text = resp.choices[0].message.content or "{}"
            match = re.search(r"\{.*\}", text, re.DOTALL)
            return json.loads(match.group() if match else text)
        except Exception as e:
            last_err = e
            _mark_busy(k)
            logger.warning("chat_json attempt %d/%d failed (key …%s): %s", i + 1, attempts, k[-4:], e)
    raise last_err or RuntimeError("All Groq keys exhausted")

def label() -> str:
    return f"Groq ({_default_model()})"

# ── complete (free-form text, no JSON) ─────────────────────────────────────

def complete(system: str, user: str) -> str:
    keys_list = keys()
    attempts = min(4, len(keys_list))
    last_err: Exception | None = None
    for i in range(attempts):
        k = _next_key()
        c = client_for(k)
        try:
            resp = c.chat.completions.create(
                model=_default_model(),
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.7,
                max_tokens=_default_max_tokens(),
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            last_err = e
            _mark_busy(k)
            logger.warning("complete attempt %d/%d failed: %s", i + 1, attempts, e)
    raise last_err or RuntimeError("All Groq keys exhausted")

# ── streaming chat (for progressive feedback) ───────────────────────────────

def stream_chat(system: str, user: str, *, max_tokens: int | None = None) -> Generator[str, None, None]:
    """Yields SSE 'data:' lines with partial text. Caller wraps in StreamingResponse."""
    keys_list = keys()
    attempts = min(4, len(keys_list))
    last_err: Exception | None = None
    for i in range(attempts):
        k = _next_key()
        c = client_for(k)
        try:
            resp = c.chat.completions.create(
                model=_default_model(),
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
                max_tokens=max_tokens or _default_max_tokens(),
                stream=True,
            )
            full = ""
            for chunk in resp:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    full += delta.content
                    yield f"data: {json.dumps({'text': full})}\n\n"
            yield f"data: {json.dumps({'done': True, 'full': full})}\n\n"
            yield "data: [DONE]\n\n"
            return
        except Exception as e:
            last_err = e
            _mark_busy(k)
            logger.warning("stream_chat attempt %d/%d failed (key …%s): %s", i + 1, attempts, k[-4:], e)
    raise last_err or RuntimeError("All Groq keys exhausted")

# ── audio transcription with failover ───────────────────────────────────────

def _reencode_to_mp3(data: bytes) -> bytes | None:
    """Re-encode raw audio to mono 16k mp3 via ffmpeg (fallback)."""
    exe = shutil.which("ffmpeg")
    if not exe:
        return None
    try:
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "input.src")
            dst = os.path.join(d, "out.mp3")
            with open(src, "wb") as fh:
                fh.write(data)
            r = subprocess.run(
                [exe, "-y", "-i", src, "-ac", "1", "-ar", "16000", "-b:a", "64k", dst],
                capture_output=True,
                timeout=30,
            )
            if r.returncode == 0 and os.path.exists(dst):
                with open(dst, "rb") as fh:
                    return fh.read()
    except Exception as e:
        logger.warning("ffmpeg re-encode failed: %s", e)
    return None

def _whisper_call(c: OpenAI, bdata: bytes, fname: str, ctype: str) -> str:
    resp = c.audio.transcriptions.create(
        model=_whisper_model(),
        file=(fname, bdata, ctype),
        response_format="json",
        temperature=0.3,
    )
    text = getattr(resp, "text", None)
    if not text:
        try:
            data = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
            text = data.get("text", "")
        except Exception:
            text = ""
    if not text or not text.strip():
        raise RuntimeError("Whisper returned an empty transcript")
    return text.strip()

def transcribe(data: bytes, mime: str = "audio/webm") -> str:
    """Transcribe audio via Groq Whisper with key failover."""
    filename = "speech." + (mime.split("/")[-1] if "/" in mime else "webm")
    keys_list = keys()
    attempts = min(4, len(keys_list))
    last_err: Exception | None = None
    for i in range(attempts):
        k = _next_key()
        c = client_for(k)
        try:
            return _whisper_call(c, data, filename, mime)
        except Exception as e:
            last_err = e
            _mark_busy(k)
            logger.warning("transcribe attempt %d/%d failed: %s", i + 1, attempts, e)
    # final fallback: re-encode to mp3 and try once more on a fresh key
    encoded = _reencode_to_mp3(data)
    if encoded:
        k = _next_key()
        c = client_for(k)
        try:
            return _whisper_call(c, encoded, "speech.mp3", "audio/mpeg")
        except Exception as e:
            last_err = e
            logger.warning("transcribe mp3 fallback failed: %s", e)
    raise last_err or RuntimeError("All Groq keys exhausted")

def whisper_label() -> str:
    return f"Groq Whisper ({_whisper_model()})"
