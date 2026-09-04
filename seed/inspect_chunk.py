import requests
r = requests.get('https://englis-edu-study.com/_next/static/chunks/553-71453236176eeb5c.js', timeout=30)
t = r.text
# Find the KEYS array definition
import re
for m in re.finditer(r'KEYS|GROQ_API_KEYS|NEXT_PUBLIC_GROQ|split', t):
    pass
# Just print a window around where env vars would be
idx = t.find('K_E_T')
# print segments
print("=== Searching for the env usage ===")
for pat in ['GROQ', 'filter', 'Boolean', 'gsk_']:
    positions = [m.start() for m in re.finditer(re.escape(pat), t)]
    print(f'{pat}: {len(positions)} occurrences')
# Show a 300-char window where "Boolean" (from .filter(Boolean)) appears
for m in re.finditer(r'Boolean', t):
    s = max(0, m.start()-100)
    print('\n--- window ---')
    print(t[s:m.start()+120])
    break
