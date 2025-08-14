import requests
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
import os
import json
import re

# 🔒 Load environment variables from .env
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama3-8b-8192"

# 📄 Download the official answer key file
file_path = hf_hub_download(
    repo_id="A2coder75/icse_board_paper_2024_physics",
    repo_type="dataset",
    filename="answer_key.json",
    token=HF_TOKEN
)

# 📚 Read the answer key into memory once
with open(file_path, "r", encoding="utf-8") as f:
    full_answer_key = json.load(f)

# 🔄 Download questions.json from Hugging Face
questions_file_path = hf_hub_download(
    repo_id="A2coder75/icse_board_paper_2024_physics",
    repo_type="dataset",
    filename="IcseXPhysicsPaper2024.json",
    token=HF_TOKEN
)
with open(questions_file_path, "r", encoding="utf-8") as f:
    questions_data = json.load(f)

# 🔍 Utility to fetch the correct answer and marks
def get_answer_for_question(section: str, question_number: str):
    section = section.strip().upper()
    question_number = question_number.strip()

    question_entry = next(
        (q for q in questions_data if q["section"].upper() == section and q["question_number"] == question_number),
        None
    )
    if not question_entry:
        return {"error": "Question not found"}

    answer_entry = next(
        (a for a in full_answer_key if a["section"].upper() == section and a["question_number"] == question_number),
        None
    )
    if not answer_entry:
        return {"error": "Answer not found"}

    return {
        "correct_answer": answer_entry["answer"],
        "marks": answer_entry["marks"],
        "question_type": question_entry.get("type", "unknown"),
        "question_text": question_entry.get("question_text", "")
    }

# 🧠 Function to evaluate a student's answer
def evaluate_answer(question_number: str, student_answer: str, section: str) -> str:
    data = get_answer_for_question(section, question_number)
    answer = data["correct_answer"]
    marks = data["marks"]
    qtype = data["question_type"]

    prompt = f"""
You are an ICSE Class 10 Physics board examiner.

Use the **official answer key ONLY** to evaluate the student’s response. Do **NOT use any external physics knowledge** or assumptions.

Answer Key:
{answer}

---

Evaluate this student's answer:

Question Number: {question_number}  
Student's Answer: {student_answer}

Instructions:
1. The question type is {qtype}
2. The total marks is {marks}

2. For Multiple Choice Questions (MCQs) or Objective questions:
   - Extract the correct option letter (a/b/c/d) and its text.
   - Normalize both the student's and correct answers by:
     - Lowercasing
     - Trimming whitespace
     - Removing parentheses
   - Award **1 mark** if:
     - Student selected the correct option letter OR
     - Student wrote exactly the correct option text
   - Award 0 if neither match.

3. For descriptive or numerical answers:
   - Check if the **meaning** of the student's answer is the same as the official key.
   - Award marks if the core **concepts**, **reasoning**, and **key words** are all present.
   - Use strict matching when formulas, values, units, or calculations are involved.
   - Award no marks if the numerical value is not exactly the same.

4. Output format:

📘 AI-Graded Evaluation:
1. Marks out of {marks}: <number>    
2. What is missing or wrong (Omit if full marks are awarded):
- <bullet points>
3. Final feedback:
<one-line helpful summary — omit if answer is perfect>
"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 512
    }

    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        return extract_json(response.json()["choices"][0]["message"]["content"]).strip()
    except Exception as e:
        return f"❌ Error: {str(e)}\n\n{response.text}"

# 🔒 Utility function to safely extract JSON from LLM response
def extract_json(text):
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return match.group(0)
    else:
        raise ValueError("No valid JSON object found.")

# 📦 Batch evaluation
def evaluate_answer_batch(batch: list) -> str:
    prompt_sections = []

    for item in batch:
        section = item["section"]
        question_number = item["question_number"]
        student_answer = item["student_answer"]

        data = get_answer_for_question(section, question_number)
        if "error" in data:
            prompt_sections.append(f"❌ Question {question_number} Error: {data['error']}")
            continue

        correct_answer = data["correct_answer"]
        marks = data["marks"]
        qtype = data["question_type"]
        question = data["question_text"]

        prompt_sections.append(f"""
--------------------------

📘 Question Number: {question_number} | Section: {section}
Question: {question}
🧠 Question Type: {qtype}  
🔢 Total Marks: {marks}

✅ **Answer Key**:  
{correct_answer}

✍️ **Student Answer**:  
{student_answer}
""")

    question_blocks = "\n".join(prompt_sections)

    full_prompt = f"""
You are an ICSE Class 10 Physics board examiner.

Use the **official answer key ONLY** to evaluate each student’s response.

Rules:
1. For MCQ/Objective:
   - Award full marks if the option letter OR the text matches (case-insensitive, trim spaces/parentheses).
   - Ignore punctuation and minor spelling mistakes.
   - Award 0 only if neither the letter nor meaning matches.

2. For Numericals:
   - Award full marks ONLY if the numerical value matches exactly within a strict tolerance:
     * If the correct answer ≤ 10: difference must be ≤ 0.005
     * If the correct answer > 10: difference must be ≤ 0.01
   - Units must match exactly or be equivalent (e.g., `m/s` vs `ms^-1`).
   - If the method is correct but the numerical is outside tolerance, award 0 marks unless partial marks are specified in the question paper.
   - Do NOT award marks for approximately correct answers unless within tolerance.


3. For Descriptive/Short Answer:
   - Award marks for **conceptual correctness** even if wording differs from the key.
   - Accept synonyms or paraphrased expressions of the same meaning.
   - If the question has multiple subparts or expects multiple points, award marks proportionally to the number of correct points.
   - Do NOT deduct marks for extra irrelevant information unless it directly contradicts the answer.

4. Output must be valid JSON with this structure:

{{
  "evaluations": [
    {{
      "question_number": "2(i)(b)",
      "section": "A",
      "question": "What type of lever is this?",
      "type": "MCQ",
      "verdict": "wrong",
      "marks_awarded": 0,
      "mistake": "Class I Lever was wrong identification because of wrong concept",
      "correct_answer": ["Class II lever"],
      "mistake_type": "conceptual",
      "feedback": "This is a class II lever because the load lies between the fulcrum and effort. Review the lever classes."
    }},
    {{
      "question_number": "2(ii)(a)",
      "section": "A",
      "question": "State one use of a concave mirror.",
      "type": "Objective",
      "marks_awarded": 1,
      "verdict": "correct",
      "mistake": [],
      "correct_answer": "Used as a shaving mirror",
      "mistake_type": [],
      "feedback": []
    }}
  ]
}}

{question_blocks}
"""


    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
def evaluate_answer_batch(batch: list) -> str:
    prompt_sections = []
    total_possible = 0

    for item in batch:
        section = item["section"]
        question_number = item["question_number"]
        student_answer = item["student_answer"]

        data = get_answer_for_question(section, question_number)
        if "error" in data:
            prompt_sections.append(f"❌ Question {question_number} Error: {data['error']}")
            continue

        correct_answer = data["correct_answer"]
        marks = data["marks"]
        total_possible += marks
        qtype = data["question_type"]
        question = data["question_text"]

        prompt_sections.append(f"""
--------------------------

📘 Question Number: {question_number} | Section: {section}
Question: {question}
🧠 Question Type: {qtype}  
🔢 Total Marks: {marks}

✅ **Answer Key**:  
{correct_answer}

✍️ **Student Answer**:  
{student_answer}
""")

    question_blocks = "\n".join(prompt_sections)

    full_prompt = f"""
You are an ICSE Class 10 Physics board examiner.

Use the **official answer key ONLY** to evaluate each student’s response.

Rules:
1. For MCQ/Objective:
   - Award full marks if the option letter OR the text matches (case-insensitive, trim spaces/parentheses).
   - Ignore punctuation and minor spelling mistakes.
   - Award 0 only if neither the letter nor meaning matches.

2. For Numericals:
   - Award full marks ONLY if the numerical value matches exactly within a strict tolerance:
     * If the correct answer ≤ 10: difference must be ≤ 0.005
     * If the correct answer > 10: difference must be ≤ 0.01
   - Units must match exactly or be equivalent (e.g., `m/s` vs `ms^-1`).
   - If the value is outside tolerance, award 0 marks unless partial marks are explicitly stated in the marking scheme.

3. For Descriptive/Short Answer:
   - Award marks for **conceptual correctness** even if wording differs from the key.
   - Accept synonyms or paraphrased expressions of the same meaning.
   - If the question has multiple subparts or expects multiple points, award marks proportionally to the number of correct points.
   - Do NOT deduct marks for extra irrelevant information unless it directly contradicts the answer.

4. Output must be valid JSON with this structure:

{{
  "evaluations": [
    {{
      "question_number": "2(i)(b)",
      "section": "A",
      "question": "What type of lever is this?",
      "type": "MCQ",
      "verdict": "wrong",
      "marks_awarded": 0,
      "total_marks": 1,
      "mistake": "Class I Lever was wrong identification because of wrong concept",
      "correct_answer": ["Class II lever"],
      "mistake_type": "conceptual",
      "feedback": "This is a class II lever because the load lies between the fulcrum and effort. Review the lever classes."
    }},
    {{
      "question_number": "2(ii)(a)",
      "section": "A",
      "question": "State one use of a concave mirror.",
      "type": "Objective",
      "marks_awarded": 1,
      "total_marks": 1,
      "verdict": "correct",
      "mistake": [],
      "correct_answer": "Used as a shaving mirror",
      "mistake_type": [],
      "feedback": []
    }}
  ],
  "total_marks_awarded": <sum of marks_awarded>,
  "total_marks_possible": {total_possible}
}}

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

