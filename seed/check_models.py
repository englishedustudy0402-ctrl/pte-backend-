import os, requests
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

keys = [k.strip() for k in os.getenv('GROQ_API_KEYS','').split(',') if k.strip()]
key = keys[0]

resp = requests.get(
    'https://api.groq.com/openai/v1/models',
    headers={'Authorization': f'Bearer {key}'},
    timeout=15,
)
print(f'Status: {resp.status_code}')
if resp.status_code == 200:
    models = resp.json().get('data', [])
    for m in models:
        print(f"  {m['id']}  owned_by={m.get('owned_by','')}")
else:
    print(resp.text[:500])
