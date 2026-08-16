import os, sys

from dotenv import load_dotenv
from anthropic import Client

load_dotenv(override=True)

key = os.getenv("ANTHROPIC_API_KEY")
if not key:
    print("NO_KEY")
    sys.exit(2)

candidate_models = [
    os.getenv("ANTHROPIC_MODEL"),
    os.getenv("EVALUATOR_MODEL"),
    "claude-sonnet-5",
    "claude-sonnet-4-20250514",
    "claude-3-7-sonnet-20250219",
    "claude-3-5-sonnet-latest",
    "claude-3-5-haiku-latest",
]
model = next(
    (
        m.strip()
        for m in candidate_models
        if m and m.strip() and m.strip() != "claude-3-5-sonnet-20241022"
    ),
    "claude-sonnet-5",
)
client = Client(api_key=key)

try:
    # 최소 토큰으로 짧은 호출을 시도합니다.
    resp = client.messages.create(
        model=model,
        messages=[{"role": "user", "content": "ping"}],
        max_tokens=1,
    )
    print("OK")
except Exception as e:
    print("ERROR", type(e).__name__, str(e))
    sys.exit(3)
