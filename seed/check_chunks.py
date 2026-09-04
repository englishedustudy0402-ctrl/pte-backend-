import requests, re

chunks = [
    "/_next/static/chunks/webpack-020c42df345e747f.js",
    "/_next/static/chunks/fd9d1056-8d24ff03c1db56a5.js",
    "/_next/static/chunks/7023-b41795b50c053cc1.js",
    "/_next/static/chunks/main-app-881f01dbe13267cd.js",
    "/_next/static/chunks/44530001-dab78ee0dc1244db.js",
    "/_next/static/chunks/4072-292b7dcf442b498b.js",
    "/_next/static/chunks/9838-31aeb7a08079d319.js",
    "/_next/static/chunks/5235-1fed4ce94115bc3d.js",
    "/_next/static/chunks/553-71453236176eeb5c.js",
    "/_next/static/chunks/6388-0ebaff953357317e.js",
    "/_next/static/chunks/5650-366fbd37ad415bee.js",
    "/_next/static/chunks/2753-6d7da2fbcebf93cf.js",
    "/_next/static/chunks/app/layout-01aa0296d58de91b.js",
]

for c in chunks:
    try:
        r = requests.get('https://englis-edu-study.com' + c, timeout=30)
        has_key = 'gsk_iHK3iq' in r.text
        has_groq = 'api.groq.com' in r.text
        has_allam = 'allam' in r.text
        if has_key or has_groq or has_allam:
            print(f'{c.split("/")[-1]}: key={has_key} groq={has_groq} allam={has_allam} len={len(r.text)}')
    except Exception as e:
        print(f'{c}: ERROR {e}')
