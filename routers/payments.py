from fastapi import APIRouter, Depends, HTTPException, Request
from middleware.security import get_current_user, get_profile, get_supabase, log_audit, get_client_ip
from pydantic import BaseModel
import razorpay, os, hmac, hashlib
from datetime import datetime, timedelta, timezone

router = APIRouter()
PLAN_PRICE_PAISE = 4900  # ₹49

def get_rzp():
    return razorpay.Client(auth=(
        os.getenv("RAZORPAY_KEY_ID"),
        os.getenv("RAZORPAY_KEY_SECRET")
    ))

@router.post("/create-order")
async def create_order(request: Request, profile=Depends(get_profile)):
    rzp = get_rzp()
    supabase = get_supabase()
    ip = get_client_ip(request)

    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    existing = supabase.table("subscriptions")\
        .select("razorpay_order_id")\
        .eq("user_id", profile["id"])\
        .eq("status", "pending")\
        .gte("created_at", cutoff)\
        .execute()

    if existing.data:
        raise HTTPException(status_code=429, detail="Order already pending")

    order = rzp.order.create({
        "amount": PLAN_PRICE_PAISE,
        "currency": "INR",
        "receipt": f"pte_{profile['id'][:8]}_{int(datetime.now().timestamp())}",
        "notes": {"user_id": profile["id"], "plan": "pro"}
    })

    supabase.table("subscriptions").insert({
        "user_id": profile["id"],
        "razorpay_order_id": order["id"],
        "amount_paise": PLAN_PRICE_PAISE,
        "status": "pending",
        "plan": "pro"
    }).execute()

    log_audit(profile["id"], "payment_order", {"order_id": order["id"]}, ip)

    return {
        "order_id": order["id"],
        "amount": PLAN_PRICE_PAISE,
        "currency": "INR",
        "key_id": os.getenv("RAZORPAY_KEY_ID"),
        "name": "PTE Platform",
        "description": "Pro Plan — ₹49/month",
        "prefill_email": profile["email"],
        "prefill_name": profile["full_name"],
    }

class VerifyPaymentRequest(BaseModel):
    razorpay_order_id: str
    razorpay_payment_id: str
    razorpay_signature: str

@router.post("/verify")
async def verify_payment(body: VerifyPaymentRequest, request: Request, user=Depends(get_current_user)):
    supabase = get_supabase()
    ip = get_client_ip(request)

    secret = os.getenv("RAZORPAY_KEY_SECRET", "")
    message = f"{body.razorpay_order_id}|{body.razorpay_payment_id}"
    expected = hmac.new(secret.encode(), message.encode(), hashlib.sha256).hexdigest()

    if not hmac.compare_digest(expected, body.razorpay_signature):
        log_audit(user.id, "payment_failed", {}, ip)
        raise HTTPException(status_code=400, detail="Invalid signature")

    order_rec = supabase.table("subscriptions")\
        .select("*")\
        .eq("razorpay_order_id", body.razorpay_order_id)\
        .eq("user_id", user.id)\
        .single()\
        .execute()

    if not order_rec.data:
        raise HTTPException(status_code=404, detail="Order not found")

    if order_rec.data["status"] == "paid":
        return {"message": "Already activated", "plan": "pro"}

    expires_at = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()

    supabase.table("subscriptions").update({
        "razorpay_payment_id": body.razorpay_payment_id,
        "razorpay_signature": body.razorpay_signature,
        "status": "paid",
        "expires_at": expires_at
    }).eq("razorpay_order_id", body.razorpay_order_id).execute()

    supabase.table("profiles").update({
        "plan": "pro",
        "trial_ends_at": None
    }).eq("id", user.id).execute()

    log_audit(user.id, "payment_success", {"order_id": body.razorpay_order_id}, ip)

    return {"message": "Payment verified. Pro activated!", "plan": "pro", "expires_at": expires_at}

@router.get("/status")
async def payment_status(profile=Depends(get_profile)):
    return {
        "plan": profile["plan"],
        "trial_ends_at": profile.get("trial_ends_at"),
    }