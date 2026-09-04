from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional
from middleware.security import get_current_user, get_supabase
from supabase import create_client
from storage3.types import CreateSignedUploadUrlOptions
import os, re, uuid, logging

router = APIRouter()
logger = logging.getLogger(__name__)

BUCKET = "recordings"
SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

class RecordingUploadRequest(BaseModel):
    attempt_id: Optional[str] = None
    question_id: Optional[str] = None
    section: Optional[str] = None
    task: Optional[str] = None
    mime: str = "audio/webm"

class RecordingDownloadRequest(BaseModel):
    attempt_id: str

def _storage():
    """Storage client bound to the private recordings bucket (service role)."""
    return create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_ROLE_KEY"))


def _is_owner(attempt_id: str, user_id: str) -> bool:
    """True only when the attempt row exists and belongs to the user. Unknown
    or errored attempts are treated as NOT owned (fail closed)."""
    if not attempt_id or not SAFE_ID.match(attempt_id):
        return False
    try:
        res = (
            get_supabase().table("attempts")
            .select("user_id").eq("id", attempt_id).maybe_single().execute()
        )
        return bool(res and res.data and res.data.get("user_id") == user_id)
    except Exception as e:
        logger.warning("ownership check failed for attempt %s: %s", attempt_id, e)
        return False


@router.post("/upload-url")
async def create_upload_url(body: RecordingUploadRequest, user=Depends(get_current_user)):
    """Returns a signed (resumable) upload URL for the user's recording, plus the
    stable storage path. Path: recordings/{user_id}/{attempt_id_or_uuid}/audio.webm
    so recordings are isolated per user and never guessable by other tenants."""
    if body.attempt_id is not None and (
        not SAFE_ID.match(body.attempt_id) or not _is_owner(body.attempt_id, user.id)
    ):
        raise HTTPException(status_code=403, detail="Not your attempt")

    supabase = get_supabase()
    attempt_id = body.attempt_id or str(uuid.uuid4())
    key = f"{user.id}/{attempt_id}/audio.webm"

    if attempt_id != body.attempt_id:
        # Persist the storage path on the attempt so scoring can later replay it.
        try:
            supabase.table("attempts").update({"audio_path": key}).eq("id", attempt_id).execute()
        except Exception as e:
            logger.warning("could not persist audio_path on attempt %s: %s", attempt_id, e)

    st = _storage()
    try:
        signed = st.storage.from_(BUCKET).create_signed_upload_url(key, CreateSignedUploadUrlOptions(upsert="true"))
        return {
            "url": signed.get("signedUrl") or signed.get("url"),
            "token": signed.get("token"),
            "path": key,
            "bucket": BUCKET,
        }
    except Exception as e:
        logger.error("could not create upload URL: %s", e)
        raise HTTPException(status_code=500, detail="Could not create upload URL")


@router.post("/signed-url")
async def get_signed_download(body: RecordingDownloadRequest, user=Depends(get_current_user)):
    """Returns a short-lived read URL for an attempt's recording. Ownership is
    enforced: a user may only read their OWN attempt recordings."""
    if not _is_owner(body.attempt_id, user.id):
        raise HTTPException(status_code=403, detail="Not your recording")

    supabase = get_supabase()
    res = (
        supabase.table("attempts")
        .select("audio_path").eq("id", body.attempt_id).maybe_single().execute()
    )
    path = None
    if res and res.data and res.data.get("audio_path"):
        path = res.data["audio_path"]
    if not path:
        path = f"{user.id}/{body.attempt_id}/audio.webm"
    if not path.startswith(f"{user.id}/"):
        raise HTTPException(status_code=403, detail="Not your recording")

    st = _storage()
    try:
        u = st.storage.from_(BUCKET).create_signed_url(path, 300)
        return {"url": u.get("signedUrl") or u.get("url"), "path": path}
    except Exception as e:
        if "not_found" in str(e).lower() or "Object not found" in str(e).lower():
            raise HTTPException(status_code=404, detail="Recording not found yet")
        logger.error("could not create read URL for %s: %s", path, e)
        raise HTTPException(status_code=500, detail="Could not create read URL")