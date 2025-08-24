import requests
import os
import json
import re
from dotenv import load_dotenv

# Load env vars
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama3-8b-8192"

def extract_json(text):
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return match.group(0)
    else:
        raise ValueError("No valid JSON object found.")

def evaluate_answer_batch(batch: list) -> str:
    """
    Batch grading where each item already contains:
    question_number, type, marks, correct_answer, user_answer
    """

    prompt_sections = []
    total_possible = 0

    for item in batch:
        qnum = item.get("question_number", "")
        qtype = item.get("type", "")
        marks = item.get("marks", 0)
        total_possible += marks
        correct_answer = item.get("correct_answer", "")
        user_answer = item.get("user_answer", "")

        prompt_sections.append(f"""
--------------------------

📘 Question Number: {qnum}
🧠 Question Type: {qtype}
🔢 Total Marks: {marks}

✅ **Answer Key**:  
{correct_answer}

✍️ **Student Answer**:  
{user_answer}
""")

    question_blocks = "\n".join(prompt_sections)

    full_prompt = f"""
You are an ICSE Class 10 Physics board examiner.

Use the **official answer key ONLY** provided in each question to evaluate the student’s response.
Award **full, partial, or zero marks** depending on how many correct elements are present.

---

## Marking Rules with Partial Credit:

1. **MCQ / Objective**
   - If only one correct option exists: full marks if correct, 0 if wrong.
   - If the question expects multiple correct choices, award marks proportionally.

2. **Numerical**
   - Break the official answer into all required **distinct numerical values** (including units).
   - Award marks per correct value:  
     `marks_awarded = (number of correct values) / (total values) × total_marks`
   - Correct means:
     * Numerical match within tolerance (≤0.005 if ≤10, ≤0.01 if >10)
     * Units match exactly or are equivalent

3. **Descriptive / Short Answer**
   - Break the official answer into **key concepts or facts**.
   - Award proportional marks:  
     `marks_awarded = (number of correct concepts) / (total concepts) × total_marks`
   - A concept counts as correct if meaning matches (synonyms allowed).
   - Extra irrelevant text does not lose marks unless it contradicts the answer.

4. Diagram:
   - For all diagram questions, award FULL marks. Images cannot be attached and thus it has to be assumed that student did correctly. 
   I REPEAT AWARD FULL MARKS TO ALL DIAGRAM QUESTIONS REGARDLESS OF STUDENT ANSWER

   IN FEEDBACK FIELD, GIVE USEFUL FEEDBACK. TELL THE STUDENT WHERE THEY WENT WRONG.
   TELL THE STUDENT IN WHICH CHAPTER ARE THEY LACKING. EXPLAIN THE ANSWER TO THE STUDENTS IN A DETAILED MANNER AND 
   TELL THEM WHERE THEY WENT WRONG AND HOW TO FIX IT. 
   DONT JUST WRITE STUDENT WROTE THIS INSTEAD OF THIS. NO. USEFUL FEEDBACK. MAKE SURE ITS HELPFUL.

---

### Output Format:
{{
  "evaluations": [
    {{
      "question_number": "4(i)",
      "type": "numerical",
      "verdict": "partially correct",
      "marks_awarded": 2,
      "total_marks": 3,
      "mistake": "Power calculation was wrong",
      "correct_answer": ["f = 24 cm", "P = 4.16 D"],
      "mistake_type": "numerical",
      "feedback": "Focal length correct, but power was miscalculated."
    }}
  ],
  "total_marks_awarded": <sum>,
  "total_marks_possible": {total_possible}
}}
RETURN THE JSON ONLY. NOTHING ELSE. I REPEAT NOTHING ELSE. NO TEXT LIKE "HERES YOUR JSON". NO. NEVER. ONLY THE JSON. STRICTLY THE JSON. NOT A CHARACTER OUTSIDE OF IT
Now evaluate these answers:
{question_blocks}
"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": full_prompt}],
        "temperature": 0.2,
        "max_tokens": 2048
    }

    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        return extract_json(response.json()["choices"][0]["message"]["content"]).strip()
    except Exception as e:
        return f"❌ Error: {str(e)}\n\n{response.text}"


