import requests, re
r = requests.get('https://englis-edu-study.com/exam/speaking/repeat', timeout=30)
print('Status:', r.status_code)
html = r.text
chunks = re.findall(r'/_next/static/chunks/[^"\' ]+\.js', html)
print('Found', len(chunks), 'chunks')
seen = set()
for c in chunks:
    if c not in seen:
        seen.add(c)
        print(' ', c)
