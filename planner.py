import os
from dotenv import load_dotenv
import requests
from typing import List

# 🌱 Load environment variables
from dotenv import dotenv_values
from io import StringIO

# Parse multi-line TOKENS env secret
tokens_raw = os.getenv("TOKENS")
if tokens_raw:
    env_vars = dotenv_values(stream=StringIO(tokens_raw))
    GROQ_API_KEY = env_vars.get("GROQ_API_KEY")
else:
    raise EnvironmentError("TOKENS secret not found in environment.")
LLAMA_MODEL = "llama3-8b-8192"  # Using the selected Llama model
DEEPSEEK_MODEL = "deepseek-r1-distill-llama-70b"

# Function to create the planner prompt based on user input
def create_planner_prompt(req) -> str:
    # Collect the relevant details from the request
    subjects = ', '.join(req.subjects)
    chapters = ', '.join(req.chapters)
    strengths = ', '.join(req.strengths)
    weaknesses = ', '.join(req.weaknesses)
    days_until_target = req.days_until_target
    time_available = req.time_available
    days_per_week = req.days_per_week
    start_date = req.start_date  # New field expected in the request

    # Prepare the output text
    prompt = f"""
You are an expert study planner for ICSE Class 10 students.

Generate a personalized study schedule based on the following inputs:

Subjects: {subjects}
Chapters: {chapters}
Study goals: {req.study_goals}
Strengths: {strengths}
Weaknesses: {weaknesses}
Start date: {req.start_date} (YYYY/MM/DD format)
Target date: {req.target} (YYYY/MM/DD format)
Time available per day: {req.time_available} hours
Days until the exam: {req.days_until_target}
Days available to study in a week: {req.days_per_week}

Create a study plan that:
- Begins from the start_date and ends at the target_date.
- Distributes subjects and chapters realistically based on {req.days_per_week} study days per week.
- Ensures that the dates in the plan match only the number of days available to study in a week.
- If the full syllabus can be covered before the target date, do so, and allocate remaining time for smart revision and buffer.
- Adds a 20-minute break between any two consecutive sessions to avoid burnout.

Return ONLY a **pure JSON object** in this exact format:
DO NOT INCLUDE ANY TEXT IN THE BEGINNING OR THE END. ALL I SHOULD GET IS A JSON WITHIN A STRING
```json
{{
  "target_date": "2025-06-15",
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
        }},
        {{
          "date": "2025-04-16",
          "tasks": [
            {{
              "subject": "Chemistry",
              "chapter": "Periodic Table",
              "task_type": "learning",
              "estimated_time": 80,
              "status": "pending"
            }},
            {{
              "break": 20
            }},
            {{
              "subject": "Physics",
              "chapter": "Work, Power, Energy",
              "task_type": "revision",
              "estimated_time": 70,
              "status": "pending"
            }}
          ]
        }}
      ]
    }}
  ]
}}
Each study day should include 1–3 tasks depending on available time, and must include a 20-minute break between two sessions. Ensure that all scheduled dates fall only on the available study days per week. """
    return prompt



# Function to interact with the Groq API for generating planner
def ask_groq_api(prompt: str, model: str) -> str:
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

