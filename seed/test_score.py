import os, requests, json
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
keys = [k.strip() for k in os.getenv('GROQ_API_KEYS','').split(',') if k.strip()]
key = keys[0]

ref = "The library is open until midnight."
hyp = "The library is open until midnight"

system = (
    'You are a senior PTE Academic speaking examiner. Score the '
    'student\'s TRANSCRIPT strictly like the real test for the given '
    'task (word-level errors for Read Aloud, correct sequence for '
    'Repeat Sentence). Return ONLY JSON, no markdown, with exactly this shape: '
    '{"transcript": "", "content": 0, "fluency": 0, "pronunciation": 0, '
    '"errors": [], "tip": "", "model": ""}. '
    'All scores 0-90.'
)

user = (
    f'Task: repeat_sentence\n'
    f'Reference: {ref}\n'
    f'Transcript: {hyp}\n'
)

for model in ['allam-2-7b', 'openai/gpt-oss-20b']:
    resp = requests.post(
        'https://api.groq.com/openai/v1/chat/completions',
        headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'},
        json={
            'model': model,
            'temperature': 0.3,
            'max_tokens': 600,
            'messages': [
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': user},
            ],
            'response_format': {'type': 'json_object'},
        },
        timeout=30,
    )
    print(f'=== {model} status {resp.status_code} ===')
    if resp.status_code == 200:
        raw = resp.json()['choices'][0]['message']['content']
        print(raw)
    else:
        print(resp.text[:200])
    print()
