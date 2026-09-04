"""WhatsApp OTP delivery via MSG91, Meta WhatsApp Cloud API, or Twilio.

Config lives in backend/.env. Until real credentials are present every send
reports "not configured" and callers fall back to the on-screen demo code.

  --- MSG91 (recommended,India: Meta rate passed through, zero markup) ---
  MSG91_AUTHKEY                API authkey from msg91.com dashboard
  MSG91_INTEGRATED_NUMBER      your WhatsApp sender number, with country code,
                               e.g. 919876543210 (digits only)
  MSG91_TEMPLATE_NAME          approved authentication (OTP) template name
  MSG91_TEMPLATE_NAMESPACE     template namespace from MSG91/Meta
  MSG91_TEMPLATE_LANGUAGE      template language code, default "en"
  MSG91_DEFAULT_COUNTRY_CODE   used when the recipient has no country code

  --- Meta WhatsApp Cloud API ---
  WHATSAPP_ACCESS_TOKEN        permanent access token from Meta Business settings
  WHATSAPP_PHONE_NUMBER_ID     numeric sender ID from WhatsApp > API Setup
  WHATSAPP_TEMPLATE_NAME       optional approved utility template (OTP) name
  WHATSAPP_TEMPLATE_LANGUAGE   template language code, default "en"
  WHATSAPP_DEFAULT_COUNTRY_CODE  used when the recipient has no country code

  --- Twilio (fastest to try) ---
  TWILIO_ACCOUNT_SID           from twilio.com console
  TWILIO_AUTH_TOKEN            from twilio.com console
  TWILIO_WHATSAPP_FROM         WhatsApp sandbox/active number, e.g. 14155238886
                               (digits only, no +)

Priority: MSG91 if configured, otherwise Meta, otherwise Twilio, otherwise
demo fallback.
"""
import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request
import logging

logger = logging.getLogger(__name__)

GRAPH_BASE = "https://graph.facebook.com"
GRAPH_VERSION = "v21.0"
TWILIO_BASE = "https://api.twilio.com/2010-04-01/Accounts"

def _placeholder(value: str) -> bool:
    v = (value or "").strip()
    return not v or "your_" in v.lower() or "placeholder" in v.lower()

def _meta_config():
    token = os.getenv("WHATSAPP_ACCESS_TOKEN", "")
    number_id = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "")
    if _placeholder(token) or _placeholder(number_id):
        return None
    return {"token": token.strip(), "number_id": number_id.strip()}

def _twilio_config():
    sid = os.getenv("TWILIO_ACCOUNT_SID", "")
    token = os.getenv("TWILIO_AUTH_TOKEN", "")
    sender = os.getenv("TWILIO_WHATSAPP_FROM", "")
    if _placeholder(sid) or _placeholder(token) or _placeholder(sender):
        return None
    return {"sid": sid.strip(), "token": token.strip(), "from": sender.strip()}

def _msg91_config():
    authkey = os.getenv("MSG91_AUTHKEY", "")
    number = os.getenv("MSG91_INTEGRATED_NUMBER", "")
    template = os.getenv("MSG91_TEMPLATE_NAME", "")
    namespace = os.getenv("MSG91_TEMPLATE_NAMESPACE", "")
    if _placeholder(authkey) or _placeholder(number) or _placeholder(template) or _placeholder(namespace):
        return None
    return {
        "authkey": authkey.strip(),
        "number": number.strip(),
        "template": template.strip(),
        "namespace": namespace.strip(),
    }

def send_otp_whatsapp(raw_phone: str, code: str) -> dict:
    to = to_e164(raw_phone)

    msg91 = _msg91_config()
    if msg91:
        return _send_msg91(msg91, to, code)

    meta = _meta_config()
    if meta:
        return _send_meta(meta, to, code)

    twilio = _twilio_config()
    if twilio:
        return _send_twilio(twilio, to, code)

    return {
        "sent": False,
        "reason": "No WhatsApp provider configured. Add MSG91 (MSG91_AUTHKEY / MSG91_INTEGRATED_NUMBER / MSG91_TEMPLATE_NAME / MSG91_TEMPLATE_NAMESPACE), Meta (WHATSAPP_ACCESS_TOKEN / WHATSAPP_PHONE_NUMBER_ID) or Twilio (TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN / TWILIO_WHATSAPP_FROM) to backend/.env, then restart the backend.",
    }

def _send_msg91(cfg: dict, to: str, code: str) -> dict:
    """Send an OTP via MSG91's WhatsApp outbound template API.

    Endpoint (verified): POST https://api.msg91.com/api/v5/whatsapp/whatsapp-outbound-message/bulk/
    Auth header: authkey: <MSG91_AUTHKEY>
    """
    lang = os.getenv("MSG91_TEMPLATE_LANGUAGE", "en").strip() or "en"
    url = "https://api.msg91.com/api/v5/whatsapp/whatsapp-outbound-message/bulk/"
    # MSG91 expects the recipient WITHOUT the leading "+" (e.g. 9198...).
    to_digits = to.lstrip("+")
    payload = {
        "integrated_number": cfg["number"],
        "content_type": "template",
        "payload": {
            "messaging_product": "whatsapp",
            "type": "template",
            "template": {
                "name": cfg["template"],
                "language": {"code": lang, "policy": "deterministic"},
                "namespace": cfg["namespace"],
                "to_and_components": [
                    {
                        "to": [to_digits],
                        "components": {
                            "body_1": {"type": "text", "value": code},
                        },
                    }
                ],
            },
        },
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "authkey": cfg["authkey"],
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            logger.info("MSG91 WhatsApp accepted: %s", body)
            return {"sent": True, "provider": "msg91", "to": to, "body": body}
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:500]
        logger.error("MSG91 WhatsApp send HTTP %s: %s", e.code, detail)
        return {"sent": False, "provider": "msg91", "reason": f"HTTP {e.code}: {detail}"}
    except Exception as e:
        logger.error("MSG91 WhatsApp send failed: %s", e)
        return {"sent": False, "provider": "msg91", "reason": str(e)}

def _send_meta(meta: dict, to: str, code: str) -> dict:
    template = os.getenv("WHATSAPP_TEMPLATE_NAME", "").strip()
    lang = os.getenv("WHATSAPP_TEMPLATE_LANGUAGE", "en").strip() or "en"

    if template:
        payload = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "template",
            "template": {
                "name": template,
                "language": {"code": lang},
                "components": [{"type": "body", "parameters": [{"type": "text", "text": code}]}],
            },
        }
    else:
        payload = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "text",
            "text": {"body": f"Your PTE Mastery verification code is {code}. It expires in 5 minutes. Do not share it with anyone."},
        }

    url = f"{GRAPH_BASE}/{GRAPH_VERSION}/{meta['number_id']}/messages"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {meta['token']}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            logger.info("Meta WhatsApp accepted: %s", body)
            return {"sent": True, "provider": "meta", "to": to}
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:500]
        logger.error("Meta WhatsApp send HTTP %s: %s", e.code, detail)
        return {"sent": False, "provider": "meta", "reason": f"HTTP {e.code}: {detail}"}
    except Exception as e:
        logger.error("Meta WhatsApp send failed: %s", e)
        return {"sent": False, "provider": "meta", "reason": str(e)}

def _send_twilio(twilio: dict, to: str, code: str) -> dict:
    body = f"Your PTE Mastery verification code is {code}. It expires in 5 minutes. Do not share it with anyone."
    form = urllib.parse.urlencode({
        "From": f"whatsapp:+{twilio['from']}",
        "To": f"whatsapp:{to}",
        "Body": body,
    }).encode("utf-8")
    auth = base64.b64encode(f"{twilio['sid']}:{twilio['token']}".encode("utf-8")).decode("ascii")
    req = urllib.request.Request(
        f"{TWILIO_BASE}/{twilio['sid']}/Messages.json",
        data=form,
        headers={
            "Authorization": f"Basic {auth}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            status = payload.get("status", "")
            logger.info("Twilio accepted, status=%s sid=%s", status, payload.get("sid"))
            return {"sent": True, "provider": "twilio", "status": status, "to": to}
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:500]
        logger.error("Twilio send HTTP %s: %s", e.code, detail)
        return {"sent": False, "provider": "twilio", "reason": f"HTTP {e.code}: {detail}"}
    except Exception as e:
        logger.error("Twilio send failed: %s", e)
        return {"sent": False, "provider": "twilio", "reason": str(e)}

def to_e164(raw_phone: str) -> str:
    digits = "".join(ch for ch in raw_phone if ch.isdigit())
    country = os.getenv("WHATSAPP_DEFAULT_COUNTRY_CODE", "91").strip()
    if digits.startswith("00"):
        digits = digits[2:]
    if len(digits) == 10 and country:
        digits = country + digits
    if digits and digits[0] == "0":
        digits = digits[1:]
    return "+" + digits

def whatsapp_configured() -> bool:
    return bool(_msg91_config() or _meta_config() or _twilio_config())