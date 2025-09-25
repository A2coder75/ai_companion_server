from typing import List
import requests
import os

# 🌱 Load environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ✅ Replace with models actually available to your account
LLAMA_MODEL = "llama3-8b-8192"
DEEPSEEK_MODEL = "deepseek-r1-distill-llama-70b"


def create_prompt(user_prompt: str, context: List[str]) -> str:
    """Format the prompt with optional context messages."""
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
    """Send request to Groq API and return response or error details."""
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
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload
        )

        # Debug: if request failed, capture full response body
        if response.status_code != 200:
            return {
                "model": model,
                "answer": f"❌ Error {response.status_code}: {response.text}",
                "tokens_used": 0
            }

        data = response.json()

        return {
            "model": model,
            "answer": data["choices"][0]["message"]["content"].strip(),
            "tokens_used": data.get("usage", {}).get("total_tokens", 0)
        }

    except Exception as e:
        return {
            "model": model,
            "answer": f"❌ Exception: {str(e)}",
            "tokens_used": 0
        }


def solve_doubt(user_prompt: str, important: bool = False, context: List[str] = []) -> dict:
    """Main entry: pick model and solve student doubt."""
    model = DEEPSEEK_MODEL if important else LLAMA_MODEL
    prompt = create_prompt(user_prompt, context)
    return ask_groq_api(prompt, model)


# 🧪 Example test
if __name__ == "__main__":
    resp = solve_doubt("What is photosynthesis?")
    print(resp)
