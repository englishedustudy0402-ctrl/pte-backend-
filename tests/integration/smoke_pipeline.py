"""Live E2E smoke test of the production question -> attempt -> scoring pipeline.

Creates throwaway Supabase auth users, drives the real FastAPI app over
TestClient, then cleans up the rows it created. Uses backend/.env credentials.
Run:  python tests/integration/smoke_pipeline.py
"""
import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

BACKEND = Path(__file__).resolve().parents[2]
load_dotenv(BACKEND / ".env")
URL = os.environ["SUPABASE_URL"].rstrip("/")
KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

sys.path.insert(0, str(BACKEND))
from fastapi.testclient import TestClient  # noqa: E402
import main  # noqa: E402

client = TestClient(main.app, raise_server_exceptions=False)

SUFFIX = str(int(time.time()))[-6:]


def auth_headers(token):
    return {"Authorization": f"Bearer {token}"}


def create_user(tag):
    body = {"email": f"pipe.{tag}{SUFFIX}@test.local", "password": "Pipe#2026x!", "email_confirm": True}
    r = requests.post(f"{URL}/auth/v1/admin/users", headers={"apikey": KEY, "Authorization": f"Bearer {KEY}"}, json=body, timeout=60)
    r.raise_for_status()
    user = r.json()
    uid = user["id"]
    h = {"apikey": KEY, "Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
    requests.patch(
        f"{URL}/rest/v1/profiles?id=eq.{uid}",
        headers={**h, "Prefer": "return=minimal"},
        json={"plan": "pro", "trial_ends_at": "2027-01-01T00:00:00Z"},
        timeout=60,
    )
    prof = requests.get(f"{URL}/rest/v1/profiles?id=eq.{uid}&select=id,plan", headers=h, timeout=60)
    if not prof.json():
        requests.post(f"{URL}/rest/v1/profiles", headers={**h, "Prefer": "return=minimal"}, json={"id": uid, "plan": "pro"}, timeout=60)
    auth_ok = requests.post(
        f"{URL}/auth/v1/token?grant_type=password",
        headers={"apikey": KEY, "Content-Type": "application/json"},
        json={"email": body["email"], "password": body["password"]},
        timeout=60,
    )
    auth_ok.raise_for_status()
    return uid, auth_ok.json()["access_token"]


def cleanup(uids):
    for uid in uids:
        h = {"apikey": KEY, "Authorization": f"Bearer {KEY}"}
        for table in ("attempts", "user_statistics"):
            requests.delete(f"{URL}/rest/v1/{table}?user_id=eq.{uid}", headers=h, timeout=60)
        requests.delete(f"{URL}/rest/v1/profiles?id=eq.{uid}", headers=h, timeout=60)
        try:
            r = requests.delete(f"{URL}/auth/v1/admin/users/{uid}", headers=h, timeout=60)
            if r.status_code not in (200, 204):
                print(f"  admin delete user {uid} -> {r.status_code} {r.text[:120]}")
        except Exception as e:
            print(f"  admin delete failed: {e}")


def main():
    uids = []
    created_attempt_ids = []
    try:
        uid, token = create_user("a")
        uids.append(uid)
        H = auth_headers(token)

        # 1) question delivery
        r = client.get("/questions/next", params={"section": "reading", "task": "mcq", "index": 400}, headers=H)
        assert r.status_code == 200, r.text
        q = r.json()
        assert q["tier"] == "premium" and q["task"] == "mcq", q
        print(f"[1] /questions/next reading/mcq#400 -> ok ({q['section']}/{q['task']} {q['question_key']})")

        r = client.get("/questions/", params={"section": "reading", "task": "mcq", "limit": 5}, headers=H)
        assert r.status_code == 200 and len(r.json()) > 0
        assert "content" not in r.json()[0], "bulk listing must not leak content"
        print(f"[1b] /questions listing metadata-only -> ok ({len(r.json())} rows)")

        r = client.get(f"/questions/{q['id']}", headers=H)
        assert r.status_code == 200 and r.json()["id"] == q["id"]
        print(f"[1c] /questions/{{{q['id'][:8]}}}... -> ok")

        # 2) attempt lifecycle + scoring (multiple task types)
        scenarios = [
            ("reading", "mcq", 3, "mcq"),
            ("reading", "reorder", 2, "reorder"),
            ("reading", "fill_blanks", 1, "fill_blanks"),
            ("listening", "dictation", 1, "dictation"),
            ("speaking", "answer_short", 4, "answer_short"),
        ]
        for section, task, qk, label in scenarios:
            created = client.post("/attempts/", json={"section": section, "task": task, "question_key": qk}, headers=H)
            assert created.status_code == 200, created.text
            aid = created.json()["attempt_id"]
            created_attempt_ids.append(aid)

            question = created.json()["question"]
            if task == "mcq":
                answer = json.dumps([question["content"]["correct"][0]])
            elif task == "reorder":
                answer = json.dumps([0, 1, 2, 3, 4])
            elif task == "fill_blanks":
                blanks = question["content"]["blanks"]
                answer = json.dumps({k: v["correct"] for k, v in blanks.items()})
            elif task == "dictation":
                answer = question["content"]["text"]
            else:
                ad = question.get("answer_data") or {}
                answer = (
                    ad.get("answer")
                    or ad.get("correct_answers")
                    or question["content"].get("answer")
                    or ""
                )
                if isinstance(answer, list):
                    answer = answer[0]
            sub = client.post(f"/attempts/{aid}/submit", json={"answer_text": answer}, headers=H)
            assert sub.status_code == 200, sub.text
            res = sub.json()
            assert res["score"] is not None and res["engine"] == "deterministic" and res["engine_version"]
            print(f"[2] {section}/{task}#{qk} attempt->submit score={res['score']} comps={[c['component'] for c in res['components']]}")

            # duplicate submit -> 409
            again = client.post(f"/attempts/{aid}/submit", json={"answer_text": answer}, headers=H)
            assert again.status_code == 409, again.text
            print(f"[2b] duplicate submit {section}/{task} -> 409 ok")

            detail = client.get(f"/attempts/{aid}", headers=H)
            assert detail.status_code == 200 and detail.json()["score"]["overall_score"] == res["score"]
            assert detail.json()["score"]["components"], "score components must be persisted"
            print(f"[2c] GET /attempts/{{id}} shows persisted score+components ok")

            fb = client.post(f"/attempts/{aid}/feedback", headers=H)
            assert fb.status_code == 200, fb.text
            assert "ai_engine" in fb.json()
            print(f"[2d] feedback endpoint {section}/{task} -> ai_engine={fb.json()['ai_engine']} ok")

        my = client.get("/attempts/my", params={"limit": 10}, headers=H)
        assert my.status_code == 200 and len(my.json()) == len(scenarios)
        print(f"[3] /attempts/my -> {len(my.json())} attempts with scores ok")

        # 4) IDOR: a second user cannot read the first user's attempt
        uid2, token2 = create_user("b")
        uids.append(uid2)
        H2 = auth_headers(token2)
        other = client.get(f"/attempts/{created_attempt_ids[0]}", headers=H2)
        assert other.status_code == 404, other.text
        print(f"[4] IDOR cross-user attempt read -> 404 ok")

        # 5) free user cannot fetch premium bank question via /questions/next
        requests.patch(
            f"{URL}/rest/v1/profiles?id=eq.{uid}",
            headers={"apikey": KEY, "Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
            json={"plan": "free", "trial_ends_at": None},
            timeout=60,
        )
        free_ok = requests.post(
            f"{URL}/auth/v1/token?grant_type=password",
            headers={"apikey": KEY, "Content-Type": "application/json"},
            json={"email": f"pipe.a{SUFFIX}@test.local", "password": "Pipe#2026x!"},
            timeout=60,
        )
        free_token = free_ok.json()["access_token"]
        fr = client.get("/questions/next", params={"section": "listening", "task": "dictation", "index": 400}, headers=auth_headers(free_token))
        assert fr.status_code == 402, fr.text
        print("[5] free user premium request -> 402 plan gate (no content leak) ok")

        listing = client.get("/questions/", params={"section": "listening", "task": "dictation", "limit": 20}, headers=auth_headers(free_token))
        assert listing.status_code == 200, listing.text
        assert all(row["tier"] == "free" for row in listing.json()), "free user listing leaked premium metadata"
        print(f"[5b] free user metadata listing -> {len(listing.json())} free-tier rows only ok")

        # 6) analytics over the persisted rollups
        ov = client.get("/analytics/overview", headers=H)
        assert ov.status_code == 200, ov.text
        assert ov.json()["total_attempts"] >= len(scenarios) and ov.json()["scored_attempts"] == len(scenarios)
        assert ov.json()["best_score"] in (0, 90)
        assert set(ov.json()["by_section"].keys()) >= {"reading", "listening", "speaking"}
        tasks = {t["task_type"] for t in ov.json()["by_task"]}
        assert {"mcq", "reorder", "fill_blanks", "dictation", "answer_short"} <= tasks
        print(f"[6] /analytics/overview -> {ov.json()['total_attempts']} attempts, {len(ov.json()['by_task'])} scored tasks ok")

        sk = client.get("/analytics/skills", headers=H)
        assert sk.status_code == 200 and sk.json()
        assert sk.json()[0]["average_score"] > 0
        print(f"[6b] /analytics/skills -> {len(sk.json())} skill rows (top={sk.json()[0]['task_type']}={sk.json()[0]['average_score']}) ok")

        tr = client.get("/analytics/trend", params={"limit": 10}, headers=H)
        assert tr.status_code == 200 and len(tr.json()["scores"]) == len(scenarios)
        print(f"[6c] /analytics/trend -> {len(tr.json()['scores'])} history points ok")

        # 7) analytics IDOR: second user's analytics are their own (empty)
        other_ov = client.get("/analytics/overview", headers=H2)
        assert other_ov.status_code == 200 and other_ov.json()["total_attempts"] == 0
        print("[7] analytics scoped per-user (b sees only their own) ok")

        print("\nALL PIPELINE SMOKE CHECKS PASSED")
    finally:
        # cleanup created attempt rows first (so roles/scories can't linger)
        h = {"apikey": KEY, "Authorization": f"Bearer {KEY}"}
        for aid in created_attempt_ids:
            requests.delete(f"{URL}/rest/v1/attempts?id=eq.{aid}", headers=h, timeout=60)
        cleanup(uids)


if __name__ == "__main__":
    main()