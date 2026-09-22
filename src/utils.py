# src/utils.py

import os
import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# Find project root and load .env
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("GROQ_API_KEY")

print("GROQ KEY LOADED:", bool(api_key))

client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)


def call_llm(prompt, max_tokens=500):
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        max_tokens=max_tokens,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content


# def call_llm_json(prompt, max_tokens=1000):
#     """Forces JSON output, parses it, returns a dict/list"""

#     text = call_llm(
#         prompt + "\n\nRespond with ONLY valid JSON, no other text.",
#         max_tokens
#     )

#     text = text.strip().replace("```json", "").replace("```", "")

#     return json.loads(text)
def call_llm_json(prompt, max_tokens=500):
    text = call_llm(
        prompt + "\n\nRespond with ONLY valid JSON, no other text.",
        max_tokens
    )

    print("RAW RESPONSE:", repr(text))

    text = text.strip().replace("```json", "").replace("```", "")

    return json.loads(text)