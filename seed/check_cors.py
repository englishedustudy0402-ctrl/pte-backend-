import requests

# Check CORS preflight for Groq with Authorization header
origin = "https://englis-edu-study.com"
url = "https://api.groq.com/openai/v1/chat/completions"

resp = requests.options(
    url,
    headers={
        "Origin": origin,
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "authorization,content-type",
    },
    timeout=20,
)
print(f"Preflight status: {resp.status_code}")
print("Access-Control-Allow-Origin:", resp.headers.get("Access-Control-Allow-Origin"))
print("Access-Control-Allow-Headers:", resp.headers.get("Access-Control-Allow-Headers"))
print("Access-Control-Allow-Methods:", resp.headers.get("Access-Control-Allow-Methods"))
print()
print("Full response headers:")
for k, v in resp.headers.items():
    print(f"  {k}: {v}")
