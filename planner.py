import os
from dotenv import load_dotenv
import requests
from typing import List

# 🌱 Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLAMA_MODEL = "llama3-8b-8192"  # or your chosen Llama model

# Function to create the planner prompt based on user input
def create_planner_prompt(req) -> str:
    subjects = ', '.join(req.subjects)
    chapters = ', '.join(req.chapters)
    strengths = ', '.join(req.strengths)
    weaknesses = ', '.join(req.weaknesses)

    prompt = f"""
You are an expert ICSE Class 10 study planner.

Generate a **personalized study plan** strictly in JSON format.

Inputs:
Subjects: {subjects}
Chapters: {chapters}
Study goals: {req.study_goals}
Strengths: {strengths}
Weaknesses: {weaknesses}
Start date: {req.start_date} (YYYY/MM/DD)
Target date: {req.target} (YYYY/MM/DD)
Time available per day: {req.time_available} hours
Days available per week: {req.days_per_week}

Requirements:
1. **Real Calendar Weeks**:  
   - Week 1 = week containing the start_date (Monday → Sunday)  
   - Week 2 = following calendar week, etc.
2. **Schedule only on allowed study days** (as per `days_per_week`).  
3. **Chronological Order**:  
   - Days in each week must be sorted by date.  
   - Tasks within a day should follow the order they should be done.  
4. Each study day can have 1–3 tasks based on available time, with a **20-minute break** between consecutive tasks:
   {{ "break": 20 }}
5. Prioritize weaker subjects first, then strengths.  
6. If the syllabus completes early, allocate remaining time for revision or buffer.  
7. **Strict JSON Output** only. No extra text.

JSON Format Example:
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
- Weeks correspond to real calendar weeks (Monday → Sunday).  
- Days are in chronological order within each week.  
- Tasks are ordered as they should be completed.  
- Only schedule tasks on allowed study days.  
- Return only JSON. No extra text.
"""
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
