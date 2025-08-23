import re
import json
import requests


# ---------------- API ----------------
def ask_groq_api(prompt: str) -> str:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer YOUR_API_KEY", "Content-Type": "application/json"}
    payload = {
        "model": "llama-3.1-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }
    res = requests.post(url, headers=headers, json=payload)
    res.raise_for_status()
    return res.json()["choices"][0]["message"]["content"]


# ---------------- Prompt ----------------
def create_planner_prompt(subject: str, topics: list[str]) -> str:
    return f"""
You are a study planner AI. Create a detailed 4-week calendar study plan.

- Subject: {subject}
- Topics: {topics}

STRICT FORMAT:
Output only valid JSON object with this schema:

{{
  "calendar_weeks": [
    {{
      "week": 1,
      "days": [
        {{
          "day": "Monday",
          "tasks": ["Task 1", "Task 2"]
        }}
      ]
    }}
  ]
}}

RULES:
- Do NOT include any text before or after JSON.
- The first character MUST be '{{' and the last MUST be '}}'.
- No explanations, no commentary.
"""


# ---------------- JSON Handling ----------------
def _extract_json(text: str) -> str:
    """Extract the largest JSON object from a text blob."""
    match = re.search(r"\{(?:[^{}]|(?R))*\}", text, re.DOTALL)
    if match:
        return match.group(0)
    raise ValueError("❌ No JSON object found in model output")


def validate_and_fix_calendar_weeks(data: dict) -> dict:
    """Ensure plan has correct structure."""
    if "calendar_weeks" not in data or not isinstance(data["calendar_weeks"], list):
        raise ValueError("Invalid plan structure")

    fixed_weeks = []
    for i, week in enumerate(data["calendar_weeks"], start=1):
        fixed_week = {
            "week": week.get("week", i),
            "days": []
        }
        for day in week.get("days", []):
            fixed_day = {
                "day": day.get("day", "Unknown"),
                "tasks": day.get("tasks", [])
            }
            fixed_week["days"].append(fixed_day)
        fixed_weeks.append(fixed_week)

    return {"calendar_weeks": fixed_weeks}


# ---------------- Main ----------------
def get_plan(subject: str, topics: list[str]) -> dict:
    prompt = create_planner_prompt(subject, topics)
    raw = ask_groq_api(prompt)

    try:
        json_str = _extract_json(raw)
        parsed = json.loads(json_str)
    except Exception as e:
        print("⚠️ JSON parse failed:", e)
        print("Raw output was:\n", raw)
        raise

    fixed = validate_and_fix_calendar_weeks(parsed)
    print("✅ Parsed study plan:")
    print(json.dumps(fixed, indent=2))
    return fixed


