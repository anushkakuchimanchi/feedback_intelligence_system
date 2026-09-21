# src/utils.py
from openai import OpenAI
import os
import json

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

def call_llm(prompt, max_tokens=1000):
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def call_llm_json(prompt, max_tokens=1000):
    """Forces JSON output, parses it, returns a dict/list"""
    text = call_llm(prompt + "\n\nRespond with ONLY valid JSON, no other text.", max_tokens)
    text = text.strip().replace("```json", "").replace("```", "")
    return json.loads(text)