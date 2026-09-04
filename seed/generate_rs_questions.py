"""
Generate PTE Repeat Sentence questions at target difficulty levels.

Targets:
  - 150 easy   (4-8 words)
  - 250 medium  (8-12 words)
  - 100 hard    (12-16 words)

Uses Groq API to generate natural, exam-realistic sentences.
Reads the existing questions.json, keeps what fits, replaces what doesn't.
"""
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# ── Groq config ───────────────────────────────────────────────────
KEYS = [k.strip() for k in os.getenv("GROQ_API_KEYS", "").split(",") if k.strip()]
BASE_URL = "https://api.groq.com/openai/v1"

# Models in preference order — allam-2-7b returns clean JSON without markdown fences
MODELS = ["allam-2-7b", "openai/gpt-oss-20b", "qwen/qwen3.6-27b"]

def next_key():
    return random.choice(KEYS)

# ── Targets ───────────────────────────────────────────────────────
TARGETS = {
    "easy":   {"count": 325, "min_words": 4,  "max_words": 8},
    "medium": {"count": 150, "min_words": 8,  "max_words": 11},
    "hard":   {"count": 25,  "min_words": 11, "max_words": 15},
}

CATEGORIES = [
    "university life", "campus facilities", "academic deadlines",
    "library services", "student services", "course requirements",
    "lecture announcements", "lab instructions", "exam procedures",
    "graduation requirements", "enrollment policies", "financial aid",
    "student organizations", "research projects", "internship programs",
    "academic integrity", "study abroad", "career services",
    "health services", "housing policies", "dining services",
    "athletic programs", "alumni events", "community outreach",
]

PTE_TIPS = [
    "Focus on meaning, not individual words. Echo each chunk as you hear it.",
    "Start speaking within 1 second of the tone. Do not wait to think.",
    "If you miss a word, skip it and keep going. Do not pause.",
    "Repeat the sentence in your head once before speaking aloud.",
    "Pay attention to numbers, dates, and proper nouns. They are memory traps.",
    "Speak at a natural pace. Rushing causes more errors than speaking slowly.",
    "Chunk the sentence by meaning groups: subject, verb, object.",
    "If the sentence has a list, remember the first and last items.",
    "Practice with a pen. Write key words while listening to train recall.",
    "Record yourself and compare with the original to spot gaps.",
]

# ── Groq API call ─────────────────────────────────────────────────

def groq_generate(prompt: str, retries=2) -> str:
    for attempt in range(retries):
        model = MODELS[attempt % len(MODELS)]
        key = next_key()
        try:
            resp = requests.post(
                f"{BASE_URL}/chat/completions",
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "temperature": 0.8,
                    "max_tokens": 2000,
                    "messages": [
                        {"role": "system", "content": "Return ONLY valid JSON. No markdown fences. No thinking. No explanation."},
                        {"role": "user", "content": prompt},
                    ],
                },
                timeout=30,
            )
            if resp.status_code == 429:
                print(f"    {model} rate limited, trying next model...")
                time.sleep(0.5)
                continue
            if resp.status_code == 400:
                err = resp.json().get("error", {}).get("message", "")
                if "thinking" in err.lower() or "enable_thinking" in err.lower():
                    # qwen needs enable_thinking=false
                    resp = requests.post(
                        f"{BASE_URL}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": model,
                            "temperature": 0.8,
                            "max_tokens": 2000,
                            "enable_thinking": False,
                            "messages": [
                                {"role": "system", "content": "Return ONLY valid JSON. No markdown fences. No thinking. No explanation."},
                                {"role": "user", "content": prompt},
                            ],
                        },
                        timeout=30,
                    )
                    if resp.status_code == 200:
                        return resp.json()["choices"][0]["message"]["content"]
                continue
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(1)
            else:
                raise
    return "{}"

# ── Parse LLM response ────────────────────────────────────────────

def parse_json_array(raw: str) -> list:
    """Extract a JSON array from LLM output, handling markdown fences and other noise."""
    # Strip markdown code fences
    cleaned = re.sub(r'```(?:json)?\s*', '', raw)
    cleaned = re.sub(r'```', '', cleaned)
    cleaned = cleaned.strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            # Model returned an object with a key containing the array
            for v in parsed.values():
                if isinstance(v, list):
                    return v
        return []
    except json.JSONDecodeError:
        pass
    # Try to find any JSON array in the text
    match = re.search(r'\[.*\]', cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return []

# ── Generate a batch of sentences ─────────────────────────────────

def generate_batch(difficulty: str, min_words: int, max_words: int, batch_size: int, used_sentences: set) -> list[dict]:
    prompt = f"""Generate exactly {batch_size} PTE Academic Repeat Sentence questions.
Difficulty: {difficulty}
Each sentence must be exactly {min_words}-{max_words} words long.

RULES:
- Natural, clean English like a real university lecture or announcement
- NO tongue-twisters, NO obscure vocabulary, NO made-up names
- Mix of topics: university life, deadlines, campus, academics, student services
- Each sentence must be grammatically correct and clear
- NO abbreviations, NO contractions (write "do not" not "don't")
- Include variety: statements, instructions, announcements, facts

Return a JSON array. Each object has:
  "text": "the sentence",
  "category": "pick from: university life, campus facilities, academic deadlines, library services, student services, course requirements, lecture announcements, lab instructions, exam procedures, graduation requirements, enrollment policies, financial aid, student organizations, research projects, internship programs, academic integrity, study abroad, career services"

Example:
[{{"text": "The library is open until midnight during exam week.", "category": "library services"}}]"""

    raw = groq_generate(prompt)
    parsed = parse_json_array(raw)

    results = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        text = item.get("text", "").strip()
        if not text:
            continue
        wc = len(text.split())
        if wc < min_words or wc > max_words:
            continue
        if text.lower() in used_sentences:
            continue
        results.append({
            "text": text,
            "category": item.get("category", random.choice(CATEGORIES)),
            "word_count": wc,
        })
        used_sentences.add(text.lower())
    return results

# ── Build a question object ────────────────────────────────────────

def make_question(idx: int, difficulty: str, text: str, category: str) -> dict:
    tip = random.choice(PTE_TIPS)
    return {
        "exam": "pte",
        "section": "speaking",
        "task": "repeat_sentence",
        "question_key": idx,
        "difficulty": difficulty,
        "question_text": text,
        "answer_data": {"reference_text": text},
        "content": {
            "text": text,
            "audioText": text,
            "speakTime": 15,
            "timeLimit": 15,
            "category": category,
            "pteTip": tip,
            "scoringLogic": {
                "dimensions": ["Content", "Oral Fluency", "Pronunciation"],
                "contentWeight": 0.4,
                "fluencyWeight": 0.3,
                "pronunciationWeight": 0.3,
                "partialCredit": True,
                "scoringMethod": "Exact word sequence matching.",
                "pteScaleNote": "Content scored on exact words in correct sequence.",
            },
        },
        "status": "published",
        "version": 1,
        "is_active": True,
        "tier": "free",
        "assets": [{"asset_type": "audio", "storage_path": f"speaking/repeat/{idx}.mp3", "is_public": False}],
    }

# ── Main ───────────────────────────────────────────────────────────

def main():
    questions_path = Path(__file__).parent / "questions.json"
    data = json.loads(questions_path.read_text(encoding="utf-8"))
    all_questions = data

    other_tasks = [q for q in all_questions if q["task"] != "repeat_sentence"]
    rs_questions = [q for q in all_questions if q["task"] == "repeat_sentence"]

    print(f"Current RS questions: {len(rs_questions)}")
    print(f"Other task questions (preserved): {len(other_tasks)}")

    used_sentences = set()
    kept = {"easy": [], "medium": [], "hard": []}

    for q in rs_questions:
        text = q.get("content", {}).get("text", "").strip()
        wc = len(text.split())
        diff = q.get("difficulty", "medium")
        low = text.lower()
        if low in used_sentences:
            continue
        target = TARGETS.get(diff)
        if target and target["min_words"] <= wc <= target["max_words"]:
            kept[diff].append(q)
            used_sentences.add(low)

    print("\nKept from existing:")
    for d in ["easy", "medium", "hard"]:
        t = TARGETS[d]
        print(f"  {d}: {len(kept[d])} kept (need {t['count']}, {t['min_words']}-{t['max_words']} words)")

    to_generate = {}
    for d in ["easy", "medium", "hard"]:
        need = TARGETS[d]["count"] - len(kept[d])
        to_generate[d] = max(0, need)

    print("\nWill generate:")
    for d in ["easy", "medium", "hard"]:
        print(f"  {d}: {to_generate[d]} new sentences")

    total_generate = sum(to_generate.values())
    if total_generate == 0:
        print("\nAll questions already match targets! Just re-indexing.")
    else:
        print(f"\nTotal to generate: {total_generate}")
        print("Generating via Groq API...\n")

        for diff in ["easy", "medium", "hard"]:
            count = to_generate[diff]
            if count == 0:
                continue
            t = TARGETS[diff]
            generated = []
            attempts = 0
            max_attempts = 20
            while len(generated) < count and attempts < max_attempts:
                attempts += 1
                remaining = count - len(generated)
                bs = min(15, remaining + 3)
                print(f"  [{diff}] attempt {attempts}: generating {bs} ({len(generated)}/{count})...")
                batch = generate_batch(diff, t["min_words"], t["max_words"], bs, used_sentences)
                generated.extend(batch)
                if batch:
                    print(f"    Got {len(batch)} valid sentences")
                else:
                    print(f"    Empty batch, retrying...")
                time.sleep(1.5)

            generated = generated[:count]
            for g in generated:
                kept[diff].append(make_question(0, diff, g["text"], g["category"]))
            print(f"  [{diff}] final: {len(generated)} generated")

    final_rs = []
    idx = 1
    for diff in ["easy", "medium", "hard"]:
        for q in kept[diff]:
            q["question_key"] = idx
            final_rs.append(q)
            idx += 1

    print(f"\nFinal RS questions: {len(final_rs)}")
    for diff in ["easy", "medium", "hard"]:
        qs = [q for q in final_rs if q["difficulty"] == diff]
        wcs = [len(q["content"]["text"].split()) for q in qs]
        if wcs:
            print(f"  {diff}: {len(qs)} questions, word range {min(wcs)}-{max(wcs)}, avg {sum(wcs)/len(wcs):.1f}")

    output = other_tasks + final_rs
    out_path = questions_path
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {len(output)} questions to {out_path}")

    print("\nWord count distribution:")
    all_wcs = {}
    for q in final_rs:
        wc = len(q["content"]["text"].split())
        all_wcs[wc] = all_wcs.get(wc, 0) + 1
    for wc in sorted(all_wcs.keys()):
        print(f"  {wc} words: {all_wcs[wc]} questions")


if __name__ == "__main__":
    main()
