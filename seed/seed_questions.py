"""
STEP 4b — Safe, idempotent seed of the offline bank into public.questions.

Reads backend/seed/questions.json (produced by
frontend/scripts/dump_offline_bank.ts) and upserts rows into Supabase via
PostgREST using the service-role key (bypasses RLS). Safe to re-run:

  * questions: upsert on (exam, section, task, question_key) — updates rows
    that already exist, inserts missing ones, never creates duplicates.
  * question_assets: rows are (re)created only for the questions touched in
    this run, after deleting that question_id's pre-existing asset rows.

Usage:
    python seed_questions.py                # uses backend/.env credentials
    python seed_questions.py --dry          # validate payloads only, no writes
"""
import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
SERVICE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
QUESTIONS_JSON = Path(__file__).resolve().parent / "questions.json"
BATCH = 500
AUDIO_TYPES = {"repeat_sentence", "answer_short", "retell_lecture", "respond_to_situation", "summarize_group_discussion"}


def headers(src):
    h = {
        "apikey": SERVICE_KEY,
        "Authorization": f"Bearer {SERVICE_KEY}",
        "Content-Type": "application/json",
    }
    h.update(src)
    return h


def rest(path, method="get", **kw):
    url = f"{SUPABASE_URL}/rest/v1/{path}"
    resp = requests.request(method, url, headers=headers(kw.pop("_headers", {})), timeout=120, **kw)
    if resp.status_code >= 400:
        raise RuntimeError(f"{method} {path} -> {resp.status_code}: {resp.text[:500]}")
    if resp.text:
        return resp.json()
    return []


def main():
    if "--dry" in sys.argv:
        dry = True
    else:
        dry = False
    if not SUPABASE_URL or not SERVICE_KEY:
        sys.exit("Missing SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY in backend/.env")
    if not QUESTIONS_JSON.exists():
        sys.exit(f"Missing {QUESTIONS_JSON} — run the dump script first")

    rows = json.loads(QUESTIONS_JSON.read_text(encoding="utf-8"))
    print(f"loaded {len(rows)} question rows from {QUESTIONS_JSON.name}")

    problems = []
    for r in rows:
        if not r.get("exam") or not r.get("section") or not r.get("task"):
            problems.append(f"missing exam/section/task: {r.get('task')}#{r.get('question_key')}")
        if not isinstance(r.get("question_key"), int) or not (1 <= r["question_key"] <= 500):
            problems.append(f"bad question_key: {r.get('task')}#{r.get('question_key')}")
        if not r.get("difficulty") or r["difficulty"] not in ("easy", "medium", "hard"):
            problems.append(f"bad difficulty: {r.get('task')}#{r.get('question_key')}")
        if not isinstance(r.get("answer_data"), dict):
            problems.append(f"bad answer_data: {r.get('task')}#{r.get('question_key')}")
        if not isinstance(r.get("content"), dict):
            problems.append(f"bad content: {r.get('task')}#{r.get('question_key')}")
        if r.get("tier") not in ("free", "premium"):
            problems.append(f"bad tier: {r.get('task')}#{r.get('question_key')}")
    if problems:
        print("payload problems:", problems[:20])
        sys.exit(1)

    if dry:
        print("dry-run: payloads validated, nothing written")
        return

    q_rows = [{k: r[k] for k in ("exam", "section", "task", "question_key", "difficulty",
                                 "question_text", "answer_data", "content", "status", "version",
                                 "is_active", "tier")} for r in rows]

    upserted = []
    for i in range(0, len(q_rows), BATCH):
        chunk = q_rows[i : i + BATCH]
        got = rest(
            "questions?on_conflict=exam,section,task,question_key",
            method="post",
            _headers={"Prefer": "resolution=merge-duplicates, return=representation"},
            json=chunk,
        )
        upserted.extend(got)
    print(f"questions upserted: {len(upserted)}")

    by_key = {(r["section"], r["task"], r["question_key"]): r["id"] for r in upserted}
    missing = [r for r in rows if (r["section"], r["task"], r["question_key"]) not in by_key]
    if missing:
        print("WARNING rows not returned by upsert:", [f"{m['task']}#{m['question_key']}" for m in missing][:10])

    asset_rows = []
    skipped = 0
    for r in rows:
        qid = by_key.get((r["section"], r["task"], r["question_key"]))
        if not qid:
            skipped += 1
            continue
        for a in r.get("assets", []):
            if not a.get("storage_path"):
                skipped += 1
                continue
            asset_rows.append(
                {
                    "question_id": qid,
                    "asset_type": a["asset_type"],
                    "storage_path": a["storage_path"],
                    "is_public": a.get("is_public", False),
                    "voice": a.get("voice"),
                    "duration_ms": a.get("duration_ms"),
                }
            )
    print(f"asset rows to write: {len(asset_rows)} (skipped {skipped})")

    qids = list(set(by_key.values()))
    # remove pre-existing asset rows for the questions being touched (idempotent re-run)
    for i in range(0, len(qids), 200):
        idlist = ",".join(qids[i : i + 200])
        rest(f"question_assets?question_id=in.({idlist})", method="delete")
    print(f"cleared old assets for {len(qids)} questions")

    for i in range(0, len(asset_rows), BATCH):
        rest("question_assets", method="post", json=asset_rows[i : i + BATCH])
    print(f"assets inserted: {len(asset_rows)}")

    stats = rest("questions?select=task&limit=12000")
    counts = {}
    for s in stats:
        counts[s["task"]] = counts.get(s["task"], 0) + 1
    print("total questions by task:", json.dumps(counts, sort_keys=True))

    assets = rest("question_assets?select=id&limit=20000")
    print(f"total asset rows now: {len(assets)}")


if __name__ == "__main__":
    main()