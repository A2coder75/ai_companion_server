import os
from dotenv import load_dotenv
import requests

# 🌱 Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLAMA_MODEL = "llama3-8b-8192"

def create_planner_prompt(req) -> str:
    subjects = ', '.join(req.subjects)
    chapters = ', '.join(req.chapters)
    strengths = ', '.join(req.strengths)
    weaknesses = ', '.join(req.weaknesses)

    prompt = f"""
You are an expert ICSE Class 10 study planner. Your job is to create a highly detailed, realistic study plan.

Here is all the information about the student:

Subjects: {subjects}
Chapters: {chapters}
Study goals: {req.study_goals}
Strengths: {strengths}
Weaknesses: {weaknesses}
Start date: {req.start_date} (YYYY/MM/DD)
Target date: {req.target} (YYYY/MM/DD)
Time available per day: {req.time_available} minutes
Days until the exam: {req.days_until_target}
Days available to study in a week: {req.days_per_week}

Important rules you must follow:

1. **Calendar weeks**:
   - A week starts on Monday and ends on Sunday.
   - Week 1 is the calendar week containing the start date.
   - Week 2 is the next Monday → Sunday, and so on until the target date.
   - Assign each study day to the correct week_number according to the calendar.

2. **Study days**:
   - Only include days when the student is available to study (as per `days_per_week`).
   - Tasks within each day must be in the order they should be completed.
   - Each day can have 1–3 tasks depending on available time.
   - Include a 20-minute break between tasks.

3. **Task allocation**:
   - Prioritize learning weak subjects first, then strengths.
   - Mix subjects intelligently across the week (do not repeat same subject excessively).
   - If syllabus is completed before the target date, allocate remaining days for **revision and buffer**.
   - Tasks should be realistic: estimated_time in minutes must fit the student's available time per day.

4. **Output rules**:
   - Return **ONLY a JSON object**.
   - Dates must increase chronologically and match the correct week_number.
   - Do not skip or reorder dates outside the available study days.
   - Example format (your output must follow this exactly):

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
        }},
        {{
          "date": "2025-04-17",
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
Critical instructions:

Do NOT merge tasks from different weeks into the same week_number.

Do NOT place dates out of order.

Week_number must reflect the actual calendar week.

Make sure the JSON is valid and parsable.

Your output should be ready-to-use without modification.

Generate the study plan based on these rules, making it realistic, well-distributed, and properly segmented into calendar weeks."""

return prompt"""
def ask_groq_api(prompt: str, model: str) -> str:
headers = {
"Authorization": f"Bearer {GROQ_API_KEY}",
"Content-Type": "application/json",
}
payload = {
"model": model,
"messages": [{"role": "user", "content": prompt}],
"temperature": 0.3,
"max_tokens": 3000,
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
