# src/utils.py
import os
import json
import time
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY1"),
    base_url="https://api.groq.com/openai/v1"
    # api_key=os.getenv("GEMINI_API_KEY"),
    # base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

#MODEL = "gemini-3.6-flash"
MODEL = "openai/gpt-oss-120b"

def call_llm(prompt, max_tokens=800, retries=4):
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
                extra_body={"reasoning_effort": "low"},
            )
            return response.choices[0].message.content
        except Exception as e:
            msg = str(e)
            if "429" in msg and "per day" in msg:
                raise Exception("DAILY_LIMIT reached. Rerun after the quota refills. " + msg[:120])
            if "429" in msg or "503" in msg:
                wait = (attempt + 1) * 20
                print(f"  Rate limited or busy, waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise Exception("Failed after retries")

def call_llm_json(prompt, max_tokens=800):
    text = call_llm(prompt, max_tokens)
    text = text.strip().replace("```json", "").replace("```", "")
    return json.loads(text)