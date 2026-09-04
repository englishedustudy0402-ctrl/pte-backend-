"""
Generate PTE Read Aloud passages matching real-PTE length/difficulty.

Real PTE Read Aloud: a SHORT passage, usually one sentence of roughly
10-45 words. Difficulty comes from vocabulary/complexity, not marathon length.
We keep the challenge but make it the SAME KIND as exam day.

Targets (word counts are realistic single-sentence/very-short-paragraph):
  - easy   (250): 10-20 words, clear simple sentences
  - medium (175): 18-32 words, moderate sentences + some unfamiliar words
  - hard   (  75): 28-45 words, complex/academic vocabulary + pronunciation traps

Writes directly to frontend/src/data/offlineBank/raPassages500.ts in the
format the rest of the app expects:
  { id, text, topic, difficulty }
"""
import json
import os
import random
import re
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

BACKEND = Path(__file__).resolve().parents[1]
load_dotenv(BACKEND / ".env")

FRONTEND_BANK = BACKEND / ".." / "frontend" / "src" / "data" / "offlineBank" / "raPassages500.ts"
FRONTEND_BANK = FRONTEND_BANK.resolve()

KEYS = [k.strip() for k in os.getenv("GROQ_API_KEYS", "").split(",") if k.strip()]
BASE_URL = "https://api.groq.com/openai/v1"
MODELS = ["allam-2-7b", "openai/gpt-oss-20b", "qwen/qwen3.6-27b"]

TARGETS = {
    "easy":   {"count": 250, "min": 9,  "max": 20},
    "medium": {"count": 175, "min": 17, "max": 32},
    "hard":   {"count": 75,  "min": 26, "max": 45},
}

TOPICS = [
    "Science", "Technology", "Environment", "Climate", "History", "Archaeology",
    "Psychology", "Neuroscience", "Economics", "Business", "Education", "Health",
    "Medicine", "Sociology", "Culture", "Marine Biology", "Art", "Architecture",
    "Linguistics", "Energy", "Transportation", "Agriculture", "Space", "Tourism",
    "Music", "Literature", "Geography", "Physics", "Chemistry", "Biology",
]

def next_key():
    return random.choice(KEYS)


def groq_generate(prompt, retries=2):
    for attempt in range(retries):
        model = MODELS[attempt % len(MODELS)]
        key = next_key()
        try:
            resp = requests.post(
                f"{BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "temperature": 0.85,
                    "max_tokens": 3000,
                    "messages": [
                        {"role": "system", "content": "Return ONLY valid JSON. No markdown fences. No thinking. No explanation."},
                        {"role": "user", "content": prompt},
                    ],
                },
                timeout=45,
            )
            if resp.status_code == 429:
                print(f"    {model} rate limited, trying next...")
                time.sleep(0.5)
                continue
            if resp.status_code == 400:
                err = resp.json().get("error", {}).get("message", "")
                if "thinking" in err.lower() or "enable_thinking" in err.lower():
                    resp = requests.post(
                        f"{BASE_URL}/chat/completions",
                        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                        json={
                            "model": model,
                            "temperature": 0.85,
                            "max_tokens": 3000,
                            "enable_thinking": False,
                            "messages": [
                                {"role": "system", "content": "Return ONLY valid JSON. No markdown fences. No thinking. No explanation."},
                                {"role": "user", "content": prompt},
                            ],
                        },
                        timeout=45,
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


def parse_json_array(raw):
    cleaned = re.sub(r'```(?:json)?\s*', '', raw)
    cleaned = re.sub(r'```', '', cleaned).strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            for v in parsed.values():
                if isinstance(v, list):
                    return v
        return []
    except json.JSONDecodeError:
        pass
    match = re.search(r'\[.*\]', cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return []


def generate_batch(difficulty, min_w, max_w, batch_size, used):
    prompt = f"""Generate exactly {batch_size} PTE Academic Read Aloud passages.
Difficulty: {difficulty}

Read Aloud in the real PTE exam shows ONE SHORT academic passage (normally a single sentence or two short sentences). The student reads it aloud. Difficulty comes from the VOCABULARY and PRONUNCIATION, not from long length.

Each passage must be exactly {min_w}-{max_w} words total.

RULES:
- Each passage is a natural, factual, single academic sentence (or at most two short sentences joined by punctuation on same subject).
- Natural academic English on varied topics (science, history, environment, technology, health, economics, art, psychology, education, space...).
- {"Simple, clear vocabulary every PTE test-taker knows. Short, everyday words. No difficult names." if difficulty=='easy' else ("Moderate academic vocabulary with a few less-common words. Include one or two words that are slightly tricky to pronounce." if difficulty=='medium' else "Rich academic vocabulary and words that are genuinely hard to pronounce clearly (e.g. longer/academic terms, specific numbers, dates, proper nouns with stress challenges). Natural but sophisticated sentence structure.")}
- {"Prefer simple structure: subject + verb + object." if difficulty=='easy' else "Use varied but grammatically correct structure."}
- NO tongue-twisters, NO made-up words, NO fragments.
- Do NOT use apostrophe contractions (write "it is" not "it's").
- Each passage ends with a period.

Return a JSON array. Each object: {{"text": "the passage", "topic": "one of: Science, Technology, Environment, History, Psychology, Economics, Education, Health, Culture, Space, Geography, Biology, Physics, Chemistry, Art, Music, Literature"}}

Example for easy: [{{"text": "Water freezes at zero degrees Celsius.", "topic": "Science"}}]"""

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
        if wc < min_w or wc > max_w:
            continue
        if text.rstrip('.').lower() in used:
            continue
        # Must be a single sentence (or two max). Two sentences only allowed for hard/medium.
        sentences = [s for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        if len(sentences) > 2:
            continue
        if difficulty == 'easy' and len(sentences) > 1:
            continue
        results.append({
            "text": text,
            "topic": item.get("topic", random.choice(TOPICS)),
        })
        used.add(text.rstrip('.').lower())
    return results


def gen_type(ts):
    lines = []
    idx = 1
    for diff in ["easy", "medium", "hard"]:
        t = TARGETS[diff]
        count = t["count"]
        generated = []
        used = set()
        attempts = 0
        max_attempts = 25
        while len(generated) < count and attempts < max_attempts:
            attempts += 1
            remaining = count - len(generated)
            bs = min(18, remaining + 4)
            print(f"  [{diff}] attempt {attempts}: generating {bs} (have {len(generated)}/{count})...")
            batch = generate_batch(diff, t["min"], t["max"], bs, used)
            generated.extend(batch)
            print(f"    got {len(batch)} valid")
            if batch:
                time.sleep(1.0)
            else:
                time.sleep(2.0)
        generated = generated[:count]
        for g in generated:
            lines.append(f"  {{ id: {idx}, text: {json.dumps(g['text'])}, topic: {json.dumps(g['topic'])}, difficulty: '{diff}' }},")
            idx += 1
        wcs = [len(g['text'].split()) for g in generated]
        if wcs:
            print(f"  [{diff}] final {len(generated)}; words {min(wcs)}-{max(wcs)}, avg {sum(wcs)/len(wcs):.1f}")
    return lines


def main():
    print(f"Output file: {FRONTEND_BANK}")
    print("Generating RA passages via Groq API...\n")
    lines = gen_type(None)

    header = "// PTE Academic Read Aloud passages — regenerated to match real-PTE "
    header += "\n// length and difficulty: short single-sentence passages (mostly ~10-30 words,\n// up to ~45 for the hardest) where the challenge is vocabulary & pronunciation,\n// exactly as in the real exam."
    body = "export const RA_PASSAGES_500: { id: number; text: string; topic: string; difficulty: 'easy' | 'medium' | 'hard' }[] = [\n" + "\n".join(lines) + "\n]\n"
    FRONTEND_BANK.write_text(header + "\n\n" + body, encoding="utf-8")
    print(f"\nWrote {len(lines)} passages to {FRONTEND_BANK}")


if __name__ == "__main__":
    main()
