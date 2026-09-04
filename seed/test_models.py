import os, requests, time
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

keys = [k.strip() for k in os.getenv('GROQ_API_KEYS','').split(',') if k.strip()]
key = keys[0]

models_to_try = ['qwen/qwen3.6-27b', 'openai/gpt-oss-20b', 'allam-2-7b', 'openai/gpt-oss-120b']
for m in models_to_try:
    resp = requests.post(
        'https://api.groq.com/openai/v1/chat/completions',
        headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'},
        json={
            'model': m,
            'temperature': 0.8,
            'max_tokens': 200,
            'messages': [{'role': 'user', 'content': 'Generate 3 short English sentences about university life. Return as JSON array with text and category fields.'}],
        },
        timeout=30,
    )
    status = resp.status_code
    if status == 200:
        body = resp.json()['choices'][0]['message']['content']
        print(f'{m}: WORKS')
        print(f'  Sample: {body[:300]}')
    else:
        err = resp.json().get('error', {}).get('message', '')[:120]
        print(f'{m}: {status} - {err}')
    time.sleep(1)
