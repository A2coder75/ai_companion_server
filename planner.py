import os
from dotenv import load_dotenv
import requests
from typing import List

# 🌱 Load environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLAMA_MODEL = "llama3-8b-8192"  # Using the selected Llama model
DEEPSEEK_MODEL = "deepseek-r1-distill-llama-70b"

# Function to create the planner prompt based on user input
def create_planner_prompt(req) -> str:
    subjects = ', '.join(req.subjects)
    chapters = ', '.join(req.chapters)
    strengths = ', '.join(req.strengths)
    weaknesses = ', '.join(req.weaknesses)

    prompt = f"""
You are an expert ICSE Class 10 study planner.

Generate a personalized study schedule based on these inputs:

Subjects: {subjects}
Chapters: {chapters}
Study goals: {req.study_goals}
Strengths: {strengths}
Weaknesses: {weaknesses}
Start date: {req.start_date} (YYYY/MM/DD)
Target date: {req.target} (YYYY/MM/DD)
Time available per day: {req.time_available} hours
Days until the exam: {req.days_until_target}
Days available to study in a week: {req.days_per_week}

Important instructions:
1. **Use real calendar weeks**. A week starts on Monday and ends on Sunday.  
   - Only include days when the student is available to study (as per `days_per_week`).
   - Assign tasks in chronological order within each week.
2. **Week numbering should match real weeks**:
   - Week 1 = the week containing the start date (Monday → Sunday)
   - Week 2 = the following calendar week, and so on.
3. Each study day can have 1–3 tasks depending on available time, with a 20-minute break between sessions.
4. If the syllabus completes early, allocate remaining time for smart revision or buffer.
5. Return ONLY a **JSON object**, no extra text.
6. Example format:
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
Make sure:
- Days in each week are in real chronological order.
- Tasks within a day are in the order they should be done.
- Weeks correspond to actual calendar weeks (Monday → Sunday)."""
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


