"""
Calendar-True Study Planner
- Enforces **real** Monday–Sunday calendar weeks via prompt + post-processing.
- Works with Groq's Chat Completions API (LLAMA models, e.g., llama3-8b-8192).

Usage:
  - Set GROQ_API_KEY in your environment.
  - Call get_plan(req_dict) → returns validated JSON dict.

This script does **two** things so the AI stops making week-number mistakes:
1) Strengthened prompt: forces the model to assign week_number by real calendar Monday–Sunday windows (start at week 0).
2) Validator/Regrouper: after the model returns JSON, we re-check every date and regroup them into correct calendar weeks, renumbered from week 0.
"""
from __future__ import annotations

import os
import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

import requests
from dotenv import load_dotenv

# ------------------------------
# Config
# ------------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLAMA_MODEL = os.getenv("LLAMA_MODEL", "llama3-8b-8192")
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

# ------------------------------
# Data Model
# ------------------------------
@dataclass
class StudentRequest:
    subjects: List[str]
    chapters: List[str]
    study_goals: str
    strengths: List[str]
    weaknesses: List[str]
    time_available: int
    target: List[int]  # [YYYY, M, D]
    days_until_target: int
    days_per_week: List[str]  # e.g., ["monday", "wednesday", ...]
    start_date: List[int]     # [YYYY, M, D]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StudentRequest":
        return cls(
            subjects=d["subjects"],
            chapters=d["chapters"],
            study_goals=d.get("study_goals", ""),
            strengths=d.get("strengths", []),
            weaknesses=d.get("weaknesses", []),
            time_available=int(d["time_available"]),
            target=list(d["target"]),
            days_until_target=int(d["days_until_target"]),
            days_per_week=list(d["days_per_week"]),
            start_date=list(d["start_date"]),
        )

# ------------------------------
# Date Helpers
# ------------------------------
WEEKDAY_TO_NAME = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def to_date(parts: List[int]) -> date:
    return date(parts[0], parts[1], parts[2])


def monday_of(d: date) -> date:
    return d - timedelta(days=d.weekday())  # Monday = 0


def sunday_of(d: date) -> date:
    return monday_of(d) + timedelta(days=6)


def iso(d: date) -> str:
    return d.isoformat()


# ------------------------------
# Prompt Builder
# ------------------------------

def create_planner_prompt(req: StudentRequest) -> str:
    subjects = ", ".join(req.subjects)
    chapters = ", ".join(req.chapters)
    strengths = ", ".join(req.strengths)
    weaknesses = ", ".join(req.weaknesses)

    start = to_date(req.start_date)
    target = to_date(req.target)
    wk0_start = monday_of(start)
    wk0_end = sunday_of(start)
    wk1_start = wk0_start + timedelta(days=7)
    wk1_end = wk1_start + timedelta(days=6)

    # Example dates to anchor the model in the **actual** calendar
    example_mapping = (
        f"Start date {iso(start)} falls in calendar week: {iso(wk0_start)} to {iso(wk0_end)} — this is \"week_number\": 0.\n"
        f"The next calendar week begins on {iso(wk1_start)} and ends on {iso(wk1_end)} — this is \"week_number\": 1."
    )

    # JSON example needs escaped braces inside f-string
    example_json = (
        '{\n'
        '  "target_date": "' + iso(target) + '",\n'
        '  "study_plan": [\n'
        '    {\n'
        '      "week_number": 0,\n'
        '      "days": [\n'
        '        { "date": "' + iso(start) + '", "tasks": [] }\n'
        '      ]\n'
        '    },\n'
        '    {\n'
        '      "week_number": 1,\n'
        '      "days": [\n'
        '        { "date": "' + iso(wk1_start) + '", "tasks": [] }\n'
        '      ]\n'
        '    }\n'
        '  ]\n'
        '}'
    )

    prompt = f"""
You are an expert ICSE Class 10 study planner. Generate a highly detailed, realistic plan.

STUDENT INFO
- Subjects: {subjects}
- Chapters: {chapters}
- Study goals: {req.study_goals}
- Strengths: {strengths}
- Weaknesses: {weaknesses}
- Start date (YYYY-MM-DD): {iso(start)}
- Target date (YYYY-MM-DD): {iso(target)}
- Time available per study day: {req.time_available} minutes
- Days until target: {req.days_until_target}
- Allowed study days each week (lowercase): {', '.join(req.days_per_week)}

CRITICAL: CALENDAR WEEK GROUPING (STRICT)
1) A week is **always** Monday–Sunday.
2) Use **week_number starting at 0**: week 0 = calendar week that contains the start date; week 1 = the next Monday–Sunday; etc.
3) How to assign week_number:
   - First list **all study dates** in chronological order using only the allowed weekdays.
   - For each date, determine its calendar Monday–Sunday window. Assign week_number by that window.
   - Do **not** increment week_number based on the count of study days. Only increment when the **calendar Monday** changes.
4) Sorting: Group by week_number, then sort days by date ascending. Week numbers must be contiguous 0,1,2,... (no gaps).
5) Do **not** merge dates from different Monday–Sunday windows into the same week, and do **not** split a window across multiple week_number blocks.

Anchor to the actual calendar for this student:
{example_mapping}

TASK RULES
- Use only the allowed study days per week.
- Each day may have 1–3 tasks, separated by a 20-minute break object: {{"break": 20}}.
- Estimated times must fit within the per-day time budget.
- Prioritize weaker subjects first, then strengths. Mix subjects across the week.
- If syllabus completes early, allocate revision/buffer days.

OUTPUT RULES
- Return **ONLY** a valid JSON object. No commentary.
- Dates must be chronological and correctly grouped by calendar week_number.
- Schema must follow exactly:
{example_json}
- Populate the tasks realistically for this student.

SELF-CHECK BEFORE ANSWERING (MANDATORY)
- For each week block, compute the Monday and Sunday of every date in the block. They must all be identical Monday–Sunday ranges.
- If any date is outside its block's Monday–Sunday, fix the grouping and renumber from week 0.
- Ensure study_plan weeks are sorted by week_number and days sorted by date.
"""
    return prompt

# ------------------------------
# API Call
# ------------------------------

def ask_groq_api(prompt: str, model: str = LLAMA_MODEL, temperature: float = 0.2, max_tokens: int = 3500) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set in environment.")

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = requests.post(GROQ_ENDPOINT, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()

# ------------------------------
# Post-processing: Validate & Fix Week Grouping
# ------------------------------

def _extract_json(text: str) -> str:
    """Extract the largest JSON object from a text blob."""
    m = re.search(r"\{[\s\S]*\}$", text)
    if m:
        return m.group(0)
    # Fallback: find first '{' and last '}'
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def _date_range_key(dstr: str) -> Tuple[str, str]:
    d = datetime.strptime(dstr, "%Y-%m-%d").date()
    ms = monday_of(d)
    se = sunday_of(d)
    return iso(ms), iso(se)


def validate_and_fix_calendar_weeks(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Regroup days into true Monday–Sunday weeks and renumber from 0.

    Expected input schema (from the model):
      {"target_date": "YYYY-MM-DD", "study_plan": [ {"week_number": int, "days": [ {"date": "YYYY-MM-DD", ...} ] }, ... ] }
    """
    if not isinstance(plan, dict) or "study_plan" not in plan:
        return plan

    # Collect all days and map by calendar week window
    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for wk in plan.get("study_plan", []):
        for day in wk.get("days", []):
            dstr = day.get("date")
            if not dstr:
                continue
            key = _date_range_key(dstr)
            buckets.setdefault(key, []).append(day)

    # Sort buckets by week-start (Monday)
    sorted_keys = sorted(buckets.keys(), key=lambda k: k[0])

    # Within each bucket, sort days by date
    new_plan_blocks = []
    for idx, key in enumerate(sorted_keys):
        days = sorted(buckets[key], key=lambda d: d.get("date", ""))
        new_plan_blocks.append({
            "week_number": idx,  # Renumber from 0
            "days": days,
        })

    # Replace study_plan
    plan["study_plan"] = new_plan_blocks
    return plan

# ------------------------------
# High-level helper
# ------------------------------

def get_plan(req_dict: Dict[str, Any]) -> Dict[str, Any]:
    req = StudentRequest.from_dict(req_dict)
    prompt = create_planner_prompt(req)
    raw = ask_groq_api(prompt)

    # Parse JSON safely
    try:
        parsed = json.loads(_extract_json(raw))
    except Exception as e:
        raise ValueError(f"Model did not return valid JSON: {e}\nRaw: {raw[:500]}")

    # Validate / fix calendar weeks
    fixed = validate_and_fix_calendar_weeks(parsed)
    return fixed
