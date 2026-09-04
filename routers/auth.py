from fastapi import APIRouter, Depends, HTTPException, Request
from middleware.security import get_current_user, get_profile, get_supabase, log_audit, get_client_ip, limiter
from pydantic import BaseModel, EmailStr
from datetime import datetime, timedelta, timezone
import hashlib, hmac, secrets

router = APIRouter()

# ── OTP codes ──────────────────────────────────────────────────────────
# In-memory code store with 5-minute expiry. Only a SHA-256 hash of each code
# is retained — the plaintext is never returned by the API and is only handed
# to the delivery provider at send time. Endpoints return 503 until a real
# WhatsApp/email provider is configured (never leak the code as a fallback).
OTP_STORE: dict = {}
OTP_TTL = timedelta(minutes=5)
MAX_ATTEMPTS = 5
PASSWORD_MIN_LEN = 8

def make_code() -> str:
    return f"{secrets.randbelow(900000) + 100000}"

def store_code(target: str, kind: str) -> str:
    code = make_code()
    OTP_STORE[f"{kind}:{target.lower()}"] = {
        "hash": hashlib.sha256(code.encode("ascii")).hexdigest(),
        "expires_at": datetime.now(timezone.utc) + OTP_TTL,
        "attempts": 0,
    }
    return code

def _code_matches(rec: dict, code: str) -> bool:
    return bool(rec) and hmac.compare_digest(
        rec["hash"], hashlib.sha256(str(code).strip().encode("ascii")).hexdigest()
    )

def check_code(target: str, kind: str, code: str) -> bool:
    key = f"{kind}:{target.lower()}"
    rec = OTP_STORE.get(key)
    if not rec:
        return False
    if datetime.now(timezone.utc) > rec["expires_at"]:
        OTP_STORE.pop(key, None)
        return False
    if rec["attempts"] >= MAX_ATTEMPTS:
        OTP_STORE.pop(key, None)
        return False
    if _code_matches(rec, code):
        return True
    rec["attempts"] += 1
    return False

def consume_code(target: str, kind: str, code: str) -> bool:
    key = f"{kind}:{target.lower()}"
    rec = OTP_STORE.get(key)
    if not rec:
        return False
    if datetime.now(timezone.utc) > rec["expires_at"]:
        OTP_STORE.pop(key, None)
        return False
    if _code_matches(rec, code):
        OTP_STORE.pop(key, None)
        return True
    rec["attempts"] += 1
    if rec["attempts"] >= MAX_ATTEMPTS:
        OTP_STORE.pop(key, None)
    return False

class OtpSendRequest(BaseModel):
    phone: str
    channel: str = "whatsapp"

class OtpVerifyRequest(BaseModel):
    phone: str
    code: str

class RecoverySendRequest(BaseModel):
    email: EmailStr

class RecoveryVerifyRequest(BaseModel):
    email: EmailStr
    code: str

class RecoveryResetRequest(BaseModel):
    email: EmailStr
    code: str
    password: str

@router.post("/otp/send")
@limiter.limit("5/minute")
async def send_otp(body: OtpSendRequest, request: Request):
    from services.whatsapp import send_otp_whatsapp
    target = "".join(ch for ch in body.phone if ch.isdigit())
    if len(target) < 8:
        raise HTTPException(status_code=400, detail="Invalid mobile number")
    code = store_code(target, "phone")
    if body.channel == "whatsapp":
        result = send_otp_whatsapp(target, code)
        if result.get("sent"):
            log_audit("otp", "send", {"channel": "whatsapp", "to": result.get("to")}, get_client_ip(request))
            return {"ok": True, "delivered": True, "expires_in_seconds": 300}
        log_audit("otp", "send_unavailable", {"reason": result.get("reason")}, get_client_ip(request))
    raise HTTPException(status_code=503, detail="OTP delivery is not configured")

@router.post("/otp/verify")
@limiter.limit("10/minute")
async def verify_otp(body: OtpVerifyRequest, request: Request):
    target = "".join(ch for ch in body.phone if ch.isdigit())
    if not check_code(target, "phone", body.code):
        raise HTTPException(status_code=400, detail="Incorrect or expired code")
    return {"ok": True}

@router.post("/recovery/send")
@limiter.limit("5/minute")
async def send_recovery(body: RecoverySendRequest, request: Request):
    email = body.email.lower()
    # When an email provider is wired up: store_code(email, "recovery"), deliver
    # the code to `email`, then return {"ok": True, "expires_in_seconds": 300}.
    # The code must NEVER be returned in the response (account-takeover risk).
    raise HTTPException(status_code=503, detail="Recovery email delivery is not configured")

@router.post("/recovery/verify")
@limiter.limit("10/minute")
async def verify_recovery(body: RecoveryVerifyRequest, request: Request):
    if not check_code(body.email.lower(), "recovery", body.code):
        raise HTTPException(status_code=400, detail="Incorrect or expired code")
    return {"ok": True}

@router.post("/recovery/reset")
@limiter.limit("10/minute")
async def reset_password(body: RecoveryResetRequest, request: Request):
    if len(body.password) < PASSWORD_MIN_LEN:
        raise HTTPException(status_code=400, detail=f"Password must be {PASSWORD_MIN_LEN}+ characters")
    email = body.email.lower()
    if not consume_code(email, "recovery", body.code):
        raise HTTPException(status_code=400, detail="Incorrect or expired code")
    # Update the stored credential where Supabase is available; otherwise the
    # local demo store (frontend) performs the swap using the same verified code.
    try:
        supabase = get_supabase()
        user = supabase.auth.admin.list_users()
        for u in user:
            if u.email and u.email.lower() == email:
                supabase.auth.admin.update_user_by_id(u.id, {"password": body.password})
                break
    except Exception:
        pass
    log_audit("recovery", "reset", {"email": email}, get_client_ip(request))
    return {"ok": True}

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    full_name: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

@router.post("/register")
@limiter.limit("5/minute")
async def register(body: RegisterRequest, request: Request):
    if len(body.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be 8+ characters")
    
    supabase = get_supabase()
    ip = get_client_ip(request)
    
    try:
        res = supabase.auth.sign_up({
            "email": body.email,
            "password": body.password,
            "options": {"data": {"full_name": body.full_name}}
        })
        if res.user:
            trial_end = (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()
            supabase.table("profiles").update({
                "plan": "trial",
                "trial_ends_at": trial_end
            }).eq("id", res.user.id).execute()
            log_audit(res.user.id, "register", {}, ip)
        return {"message": "Account created. Check email to verify.", "user_id": res.user.id if res.user else None}
    except Exception:
        raise HTTPException(status_code=400, detail="Registration failed - the email may already be registered.")

@router.post("/login")
@limiter.limit("10/minute")
async def login(body: LoginRequest, request: Request):
    supabase = get_supabase()
    ip = get_client_ip(request)
    try:
        res = supabase.auth.sign_in_with_password({"email": body.email, "password": body.password})
        log_audit(res.user.id, "login", {}, ip)
        return {
            "access_token": res.session.access_token,
            "refresh_token": res.session.refresh_token,
            "user": {"id": res.user.id, "email": res.user.email}
        }
    except:
        raise HTTPException(status_code=401, detail="Invalid credentials")

@router.get("/me")
async def me(profile=Depends(get_profile)):
    return {k: v for k, v in profile.items() if k not in ("razorpay_customer_id",)}

@router.post("/logout")
async def logout(user=Depends(get_current_user)):
    return {"message": "Logged out"}