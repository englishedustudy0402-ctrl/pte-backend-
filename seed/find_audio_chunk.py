import requests, re

html = requests.get('https://englis-edu-study.com/exam/speaking/repeat', timeout=30).text
chunks = sorted(set(re.findall(r'/_next/static/chunks/[^"\' ]+\.js', html)))

for c in chunks:
    try:
        r = requests.get('https://englis-edu-study.com' + c, timeout=30)
    except Exception:
        continue
    t = r.text
    if 'audioWorklet' in t or 'recorder-processor' in t:
        print(f'{c.split("/")[-1]}: worklet={("recorder-processor" in t)} audioWorklet={("audioWorklet" in t)} flush={("flush" in t)} ScriptProcessor={("ScriptProcessor" in t)} onaudioprocess={("onaudioprocess" in t)}')
