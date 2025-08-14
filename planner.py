import os
from dotenv import load_dotenv
import requests

# 🌱 Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLAMA_MODEL = "llama3-8b-8192"

def create_robust_planner_prompt(req) -> str:
    subjects = ', '.join(req.subjects)
    chapters = ', '.join(req.chapters)
    strengths = ', '.join(req.strengths)
    weaknesses = ', '.join(req.weaknesses)

    prompt = f"""
You are an expert ICSE Class 10 study planner. 

**Step-by-step instructions (follow exactly):**

1. Week 1 = the week containing the start date (Monday → Sunday). Week 2 = next calendar week, and so on.
2. Schedule **only on allowed study days** from {req.days_per_week}.
3. Assign each date to its **correct week_number** based on real calendar weeks.
4. Within each week, sort days **chronologically**.
5. Each day can have 1–3 tasks depending on available time, with a 20-minute break between tasks.
6. Prioritize weaker subjects first, then strengths.
7. If the syllabus completes early, allocate remaining time for revision or buffer.
8. **Return only JSON**, exactly like this:
```json
{{
  "target_date": "{req.target}",
  "study_plan": [
    {{
      "week_number": 1,
      "days": [
        {{
          "date": "2025-04-15",
          "tasks": [
            {{
              "subject": "Physics",
              "chapter": "Force",
              "task_type": "learning",
              "estimated_time": 90,
              "status": "pending"
            }},
            {{
              "break": 20
            }},
            {{
              "subject": "Math",
              "chapter": "Quadratic Equations",
              "task_type": "revision",
              "estimated_time": 60,
              "status": "pending"
            }}
          ]
        }}
      ]
    }}
  ]
}}
- **Week_number must increment only when moving to the next real calendar week.**
- **Do not put all days into one week.**
- **Days inside each week must be in chronological order.**
- **Tasks inside each day must follow the order they should be done.**
"""
    return prompt

def ask_groq_api(prompt: str, model: str = LLAMA_MODEL) -> str:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 2500,
    }
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"❌ Error: {str(e)}"

def generate_planner(req):
    prompt = create_robust_planner_prompt(req)
    return ask_groq_api(prompt)
