from dotenv import load_dotenv
load_dotenv()

import os
print("KEY LOADED:", os.getenv("GROQ_API_KEY") is not None)

from src.utils import call_llm
print(call_llm("Say hello in 5 words"))