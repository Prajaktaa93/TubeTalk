import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

# 1. SETUP — loads key from .env file
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found. Add it to your .env file.")
genai.configure(api_key=api_key)

# 2. RUN
print("Checking models...")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(f"FOUND: {m.name}")