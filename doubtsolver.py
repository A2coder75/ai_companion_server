from typing import List
import requests
import os
import re

# 🌱 Load environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

LLAMA_MODEL = "llama3-8b-8192"
DEEPSEEK_MODEL = "deepseek-r1-distill-llama-70b"


def create_prompt(user_prompt: str, context: List[str]) -> str:
    context_block = ""
    if context:
        context_block = "\n".join([f"Previous message: {msg}" for msg in context])

    return f"""
You are an expert ICSE doubt explainer.

You must **only** give the final explanation. **Do not add any thinking process, internal reasoning, or notes**. Your answer must start immediately with the explanation.
Explain the concept in very simple words to clear the concept
DO NOT write very long answers

{context_block}

Current Doubt:
{user_prompt}

✅ Use clear steps, bullet points, and examples. Be brief but accurate. Try explaining in a way the student can understand
🚫 DONT drag out answers or hallucinate. Be accurate and concise and explain in simple words.
"""



def ask_groq_api(prompt: str, model: str) -> dict:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 1024
    }

    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

        return {
            "model": model,
            "answer": data["choices"][0]["message"]["content"].strip(),
            "tokens_used": data["usage"]["total_tokens"]
        }

    except Exception as e:
        return {
            "model": model,
            "answer": f"❌ Error: {str(e)}",
            "tokens_used": 0
        }


def solve_doubt(user_prompt: str, important: bool = False, context: List[str] = []) -> dict:
    model = DEEPSEEK_MODEL if important else LLAMA_MODEL
    prompt = create_prompt(user_prompt, context)
    return ask_groq_api(prompt, model)

