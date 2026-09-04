import json, re, logging
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from middleware.security import get_supabase, require_role
from dotenv import load_dotenv
from services.groq_rotation import complete as groq_complete, available as groq_available

load_dotenv()
router = APIRouter()
logger = logging.getLogger(__name__)

def generate_text(prompt: str) -> str:
    """Generate question JSON via Groq — the only AI provider used."""
    if not groq_available():
        raise RuntimeError("No AI provider configured (set GROQ_API_KEYS)")
    return groq_complete(
        "You generate PTE Academic practice questions. Always respond with valid JSON only (arrays of objects), no markdown.",
        prompt,
    )

DIFFICULTY_INSTRUCTION = """
IMPORTANT: Generate HARD, exam-level questions that challenge advanced English learners.
- Use complex academic vocabulary (C1-C2 level)
- Include nuanced grammar structures
- Topics: economics, science, technology, environment, sociology, medicine, philosophy
- Passages should be dense and information-rich
- Wrong options should be plausible but clearly incorrect on careful reading
- This is for PTE Academic exam preparation — make it as close to real exam difficulty as possible
"""

def parse_json(text: str):
    try:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"```json|```", "", text).strip()
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        match2 = re.search(r'\{.*\}', text, re.DOTALL)
        if match2:
            return [json.loads(match2.group())]
        return json.loads(text)
    except Exception as e:
        logger.error(f"JSON parse error: {e} | text: {text[:200]}")
        raise HTTPException(status_code=500, detail=f"AI JSON parse error: {str(e)}")

class GenerateRequest(BaseModel):
    task: str
    count: Optional[int] = 5
    difficulty: Optional[str] = "hard"

TASK_PROMPTS = {
    "read_aloud": lambda count: f"""{DIFFICULTY_INSTRUCTION}
Generate {count} PTE Read Aloud passages. Each must be 70-90 words on complex academic topics.
Use sophisticated vocabulary, subordinate clauses, technical terms.
Return ONLY valid JSON array, no markdown:
[{{"section":"speaking","task":"read_aloud","difficulty":"hard","is_active":true,"content":{{"text":"complex academic paragraph 70-90 words","prepTime":35,"speakTime":40,"timeLimit":75}}}}]""",

    "repeat_sentence": lambda count: f"""{DIFFICULTY_INSTRUCTION}
Generate {count} PTE Repeat Sentence items. Each 10-16 words, complex grammar, academic vocabulary.
Include conditionals, passive voice, complex noun phrases.
Return ONLY valid JSON array:
[{{"section":"speaking","task":"repeat_sentence","difficulty":"hard","is_active":true,"content":{{"text":"complex academic sentence","audioText":"complex academic sentence","speakTime":15,"timeLimit":15}}}}]""",

    "describe_image": lambda count: f"""{DIFFICULTY_INSTRUCTION}
Generate {count} PTE Describe Image questions with complex charts/graphs.
Use real data-like descriptions with specific numbers, trends, comparisons.
Return ONLY valid JSON array:
[{{"section":"speaking","task":"describe_image","difficulty":"hard","is_active":true,"content":{{"imageUrl":"https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/Bar_chart_of_India%27s_exports.svg/600px-Bar_chart_of_India%27s_exports.svg.png","imageAlt":"detailed description: Bar chart showing GDP growth rates across 6 countries from 2015-2023, with specific percentages and notable trends","prepTime":25,"speakTime":40,"timeLimit":65}}}}]""",

    "retell_lecture": lambda count: f"""{DIFFICULTY_INSTRUCTION}
Generate {count} PTE Re-tell Lecture audio transcripts. Each 120-150 words, complex academic lecture style.
Topics: neuroscience, quantum physics, behavioral economics, climate science.
Return ONLY valid JSON array:
[{{"section":"speaking","task":"retell_lecture","difficulty":"hard","is_active":true,"content":{{"audioText":"full 120-150 word complex academic lecture transcript","topic":"specific academic topic","prepTime":10,"speakTime":40,"timeLimit":50}}}}]""",

    "answer_short": lambda count: f"""{DIFFICULTY_INSTRUCTION}
Generate {count} PTE Answer Short Question items. Tricky factual questions requiring specific knowledge.
Questions about science, geography, medicine, economics.
Return ONLY valid JSON array:
[{{"section":"speaking","task":"answer_short","difficulty":"hard","is_active":true,"content":{{"question":"specific knowledge question","answer":"precise answer","audioText":"specific knowledge question","timeLimit":10}}}}]""",

    "summarize": lambda count: f"""{DIFFICULTY_INSTRUCTION}
Generate {count} PTE Summarize Written Text passages. 250-300 words, dense academic text.
Topics: climate policy, neural networks, economic inequality, medical breakthroughs.
Students must summarize in ONE complex sentence (5-75 words).
Return ONLY valid JSON array:
[{{"section":"writing","task":"summarize","difficulty":"hard","is_active":true,"content":{{"passage":"250-300 word dense academic passage","timeLimit":600}}}}]""",

    "essay": lambda count: f"""{DIFFICULTY_INSTRUCTION}
Generate {count} PTE Essay prompts on controversial, complex topics requiring balanced arguments.
Topics must require critical thinking, evidence-based reasoning.
Return ONLY valid JSON array:
[{{"section":"writing","task":"essay","difficulty":"hard","is_active":true,"content":{{"prompt":"complex controversial essay question requiring balanced argument","wordMin":200,"wordMax":300,"timeLimit":1200}}}}]""",

    "mcq": lambda count: f"""{DIFFICULTY_INSTRUCTION}
Generate {count} PTE Reading Single Answer MCQ. Dense 200-250 word passage, tricky question.
Wrong options should be plausible paraphrases of passage content.
Return ONLY valid JSON array:
[{{"section":"reading","task":"mcq","difficulty":"hard","is_active":true,"content":{{"passage":"200-250 word dense academic passage","question":"inference or detail question","options":["plausible wrong A","correct answer B","plausible wrong C","plausible wrong D"],"correct":[1],"timeLimit":120}}}}]""",

    "mcq_multi": lambda count: f"""{DIFFICULTY_INSTRUCTION}
Generate {count} PTE Reading Multiple Answer MCQ. 200-250 word passage, 2-3 correct answers.
Return ONLY valid JSON array:
[{{"section":"reading","task":"mcq_multi","difficulty":"hard","is_active":true,"content":{{"passage":"200-250 word dense passage","question":"which statements are true","options":["correct A","wrong B","correct C","wrong D","correct E"],"correct":[0,2,4],"timeLimit":120}}}}]""",

    "reorder": lambda count: f"""{DIFFICULTY_INSTRUCTION}
Generate {count} PTE Re-order Paragraphs. 5 sentences forming a complex academic paragraph.
Sentences should have subtle logical connectors making order non-obvious.
Return ONLY valid JSON array:
[{{"section":"reading","task":"reorder","difficulty":"hard","is_active":true,"content":{{"sentences":[{{"id":1,"text":"sentence 1"}},{{"id":2,"text":"sentence 2"}},{{"id":3,"text":"sentence 3"}},{{"id":4,"text":"sentence 4"}},{{"id":5,"text":"sentence 5"}}],"correct":[3,1,5,2,4],"timeLimit":180}}}}]""",

    "fill_blanks": lambda count: f"""{DIFFICULTY_INSTRUCTION}
Generate {count} PTE Reading Fill in Blanks. 150-200 word passage with 4-5 blanks.
Blanks test advanced vocabulary and collocations. Wrong options differ subtly.
Return ONLY valid JSON array:
[{{"section":"reading","task":"fill_blanks","difficulty":"hard","is_active":true,"content":{{"text":"passage with {{1}} {{2}} {{3}} {{4}} blanks at appropriate positions","blanks":{{"1":{{"correct":"precise word","options":["precise word","close synonym","related word","wrong word"]}},"2":{{"correct":"correct collocation","options":["correct collocation","near miss","wrong register","incorrect form"]}},"3":{{"correct":"right term","options":["right term","similar term","wrong term","unrelated"]}},"4":{{"correct":"exact word","options":["exact word","approximate","incorrect","unrelated"]}}}},"timeLimit":120}}}}]""",

    "fill_blanks_rw": lambda count: f"""{DIFFICULTY_INSTRUCTION}
Generate {count} PTE Reading & Writing Fill in Blanks. 200-250 word passage, 5-6 blanks.
Test advanced grammar AND vocabulary simultaneously.
Return ONLY valid JSON array:
[{{"section":"reading","task":"fill_blanks_rw","difficulty":"hard","is_active":true,"content":{{"text":"passage with {{1}} {{2}} {{3}} {{4}} {{5}} blanks","blanks":{{"1":{{"correct":"word","options":["word","wrong1","wrong2","wrong3"]}},"2":{{"correct":"word","options":["word","wrong1","wrong2","wrong3"]}},"3":{{"correct":"word","options":["word","wrong1","wrong2","wrong3"]}},"4":{{"correct":"word","options":["word","wrong1","wrong2","wrong3"]}},"5":{{"correct":"word","options":["word","wrong1","wrong2","wrong3"]}}}},"timeLimit":120}}}}]""",

    "summarize_spoken": lambda count: f"""{DIFFICULTY_INSTRUCTION}
Generate {count} PTE Summarize Spoken Text lectures. Complex 90-120 second academic lectures.
Dense with information, statistics, technical terms.
Return ONLY valid JSON array:
[{{"section":"listening","task":"summarize_spoken","difficulty":"hard","is_active":true,"content":{{"audioText":"complex 90-120 second academic lecture full transcript with specific data","topic":"precise academic topic","timeLimit":600}}}}]""",

    "listening_mcq": lambda count: f"""{DIFFICULTY_INSTRUCTION}
Generate {count} PTE Listening MCQ. Complex audio with tricky comprehension questions.
Return ONLY valid JSON array:
[{{"section":"listening","task":"listening_mcq","difficulty":"hard","is_active":true,"content":{{"audioText":"complex academic audio transcript 60-90 words","question":"inference question about audio","options":["correct answer","plausible wrong","plausible wrong","plausible wrong"],"correct":[0],"timeLimit":90}}}}]""",

    "listening_fill": lambda count: f"""{DIFFICULTY_INSTRUCTION}
Generate {count} PTE Listening Fill in Blanks. Audio transcript with blanks testing listening precision.
Return ONLY valid JSON array:
[{{"section":"listening","task":"listening_fill","difficulty":"hard","is_active":true,"content":{{"audioText":"full audio transcript","text":"transcript with {{1}} {{2}} {{3}} blanks","blanks":{{"1":{{"correct":"exact word from audio"}},"2":{{"correct":"exact word from audio"}},"3":{{"correct":"exact word from audio"}}}},"timeLimit":90}}}}]""",

    "highlight_summary": lambda count: f"""{DIFFICULTY_INSTRUCTION}
Generate {count} PTE Highlight Correct Summary. Complex audio, 4 plausible summaries.
Wrong summaries should contain partial truths or subtle distortions.
Return ONLY valid JSON array:
[{{"section":"listening","task":"highlight_summary","difficulty":"hard","is_active":true,"content":{{"audioText":"complex 60-90 word academic audio","options":["accurate complete summary","partially correct but missing key point","correct topic wrong conclusion","plausible but contradicts audio"],"correct":0,"timeLimit":90}}}}]""",

    "missing_word": lambda count: f"""{DIFFICULTY_INSTRUCTION}
Generate {count} PTE Select Missing Word. Sentence cut off — choose the most logical completion.
Return ONLY valid JSON array:
[{{"section":"listening","task":"missing_word","difficulty":"hard","is_active":true,"content":{{"audioText":"academic sentence that ends abruptly before the final word","options":["correct precise word","plausible wrong","contextually wrong","grammatically wrong"],"correct":0,"timeLimit":60}}}}]""",

    "highlight_incorrect": lambda count: f"""{DIFFICULTY_INSTRUCTION}
Generate {count} PTE Highlight Incorrect Words. Audio and text differ in 2-4 subtle words.
Wrong words should be similar-sounding or same-category substitutions.
Return ONLY valid JSON array:
[{{"section":"listening","task":"highlight_incorrect","difficulty":"hard","is_active":true,"content":{{"audioText":"original audio transcript","text":"written version with subtle word substitutions that differ from audio","incorrectWords":["wrong word 1","wrong word 2"],"correctWords":["correct word 1","correct word 2"],"timeLimit":90}}}}]""",

    "dictation": lambda count: f"""{DIFFICULTY_INSTRUCTION}
Generate {count} PTE Write from Dictation sentences. Complex 12-18 word academic sentences.
Include difficult vocabulary, unusual word order, technical terms.
Return ONLY valid JSON array:
[{{"section":"listening","task":"dictation","difficulty":"hard","is_active":true,"content":{{"text":"complex 12-18 word academic sentence with sophisticated vocabulary","timeLimit":90}}}}]""",
}

@router.post("/generate")
async def generate_questions(body: GenerateRequest, _=Depends(require_role("admin"))):
    supabase = get_supabase()
    prompt_fn = TASK_PROMPTS.get(body.task)
    if not prompt_fn:
        raise HTTPException(status_code=400, detail=f"Unknown task: {body.task}. Valid tasks: {list(TASK_PROMPTS.keys())}")

    prompt = prompt_fn(body.count)
    try:
        questions = parse_json(generate_text(prompt))
        inserted = []
        for q in questions:
            q["difficulty"] = "hard"
            result = supabase.table("questions").insert(q).execute()
            if result.data:
                inserted.append(result.data[0])
        return {"generated": len(inserted), "task": body.task, "questions": inserted}
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate-all")
async def generate_all(_=Depends(require_role("admin"))):
    """Generate hard questions for all PTE task types"""
    supabase = get_supabase()
    task_counts = {
        "read_aloud": 8,
        "repeat_sentence": 12,
        "describe_image": 6,
        "retell_lecture": 4,
        "answer_short": 6,
        "summarize": 3,
        "essay": 5,
        "mcq": 5,
        "mcq_multi": 4,
        "reorder": 5,
        "fill_blanks": 6,
        "fill_blanks_rw": 6,
        "summarize_spoken": 3,
        "listening_mcq": 4,
        "listening_fill": 4,
        "highlight_summary": 3,
        "missing_word": 4,
        "highlight_incorrect": 3,
        "dictation": 6,
    }

    results = {}
    for task, needed in task_counts.items():
        try:
            existing = supabase.table("questions").select("id").eq("task", task).execute()
            have = len(existing.data)
            if have >= needed:
                results[task] = {"status": "skipped", "have": have, "needed": needed}
                continue

            to_generate = needed - have
            prompt_fn = TASK_PROMPTS.get(task)
            if not prompt_fn:
                results[task] = {"status": "error", "detail": "No prompt"}
                continue

            questions = parse_json(generate_text(prompt_fn(to_generate)))

            inserted = 0
            for q in questions:
                q["difficulty"] = "hard"
                try:
                    supabase.table("questions").insert(q).execute()
                    inserted += 1
                except Exception as e:
                    logger.error(f"Insert error for {task}: {e}")

            results[task] = {"status": "generated", "inserted": inserted, "had": have}
        except Exception as e:
            results[task] = {"status": "error", "detail": str(e)}

    return results