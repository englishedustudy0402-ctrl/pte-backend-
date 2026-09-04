import os, requests
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
keys = [k.strip() for k in os.getenv('GROQ_API_KEYS','').split(',') if k.strip()]
key = keys[0]

# Test allam-2-7b coach
resp = requests.post(
    'https://api.groq.com/openai/v1/chat/completions',
    headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'},
    json={
        'model': 'allam-2-7b',
        'temperature': 0.4,
        'max_tokens': 200,
        'messages': [
            {'role': 'system', 'content': 'Return ONLY valid JSON. No markdown.'},
            {'role': 'user', 'content': 'Return JSON with feedback, tip, encouragement fields about a PTE student.'},
        ],
        'response_format': {'type': 'json_object'},
    },
    timeout=15,
)
print(f'Coach: {resp.status_code}')
if resp.status_code == 200:
    print(resp.json()['choices'][0]['message']['content'][:300])
else:
    print(resp.text[:200])
