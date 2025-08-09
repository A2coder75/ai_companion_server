import requests
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
import os
import json
import re

# 🔒 Load environment variables from .env
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
    filename="IcseXPhysicsPaper2024.json",   # 🔁 change to your file name
    token=HF_TOKEN
)
with open(questions_file_path, "r", encoding="utf-8") as f:
    questions_data = json.load(f)

# 🔍 Utility to fetch the correct answer and marks
def get_answer_for_question(section: str, question_number: str):
    section = section.strip().upper()
    question_number = question_number.strip()

    # Validate the question
    question_entry = next(
        (q for q in questions_data if q["section"].upper() == section and q["question_number"] == question_number),
        None
    )
    if not question_entry:
        return {"error": "Question not found"}

    # Find the corresponding answer
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
    }

# 🧠 Function to evaluate a student's answer
def evaluate_answer(question_number: str, student_answer: str, section: str) -> str:
    data = get_answer_for_question(section, question_number)
    answer = data["correct_answer"]
    marks = data["marks"]
    qtype = data["question_type"]

    prompt = f"""
You are an ICSE Class 10 Physics board examiner.

Use the **official answer key ONLY** to evaluate the student’s response. Do **NOT use any external physics knowledge** or assumptions. Your job is to assess factual and conceptual correctness **based strictly on the key**.

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
   - For objective questions award marks ONLY if the answer is an exact match (case and commas doesn't matter)

3. For descriptive or numerical answers:
   - Check if the **meaning** of the student's answer is the same as the official key.
   - Award marks if the core **concepts**, **reasoning**, and **key words** are all present, even if the phrasing differs.
   - Use strict matching when formulas, values, units, or calculations are involved.
   - Give **partial marks** for partially correct responses. HOWEVER if the numerical value is not exact then no marks to be awarded, no matter how close the student answer is to the answer key. It has to be EXACT 
   - The inclusion or missing of commas in numerical does not affect the marks, The answer should be numerically the same
   - Do not award marks for things not in the answer key.

4. ✍️ Output in **this exact format** (markdown-compatible):

📘 AI-Graded Evaluation:
1. Marks out of {marks}: <number>    
2. What is missing or wrong (Omit if full marks are awarded):
- <bullet points of missing or incorrect parts>
3. Final feedback:
<one-line helpful summary — omit if answer is perfect>

⚠️ Do not hallucinate. Do not guess. Stay 100% aligned with the answer key.
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
    """Extracts the first JSON object from text."""
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return match.group(0)
    else:
        raise ValueError("No valid JSON object found.")

def evaluate_answer_batch(batch: list) -> str:
    prompt_sections = []

    for item in batch:
        section = item["section"]
        question_number = item["question_number"]
        student_answer = item["student_answer"]

        # Get answer key and metadata
        data = get_answer_for_question(section, question_number)
        if "error" in data:
            prompt_sections.append(f"❌ Question {question_number} Error: {data['error']}")
            continue

        correct_answer = data["correct_answer"]
        marks = data["marks"]
        qtype = data["question_type"]
        question = data["question_text"]

        # Create individual evaluation block
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

Use the **official answer key ONLY** to evaluate each student’s response. Do **NOT** use any external knowledge or assumptions.

For each question, strictly follow the rules below:

---

1. The question type may be MCQ, descriptive, objective, or numerical.
2. For MCQ/Objective:
   - Match either the option letter or full text, if the option is matching with the answer key award full marks(case insensitive, trim spaces/parentheses).
   - Award 1 mark if matched. Else, 0.
3. For numericals:
   - Award full marks **only if** the numerical value is exactly the same (ignore commas) (Ex - 100,000 and 100000 are the same therefore both are correct)
   - If not exact, give 0 — even if close. Check the unit for full grading
4. For descriptive:
   - Award full or partial marks if all key concepts, reasoning, and keywords match.
   - Don’t award for extra info not in key.

---

💡 Output must be a strict valid JSON array named `evaluations`, like below:
the mistake return should contain what part of the answer was wrong and why
Also, mistake types can only be conceptual (if the concept is wrongly explained) and calculation (if the numerical answer is wrong), interpretation (if the question was wrongly read or understood), 
Also include a feedback field where you explain the answer and ask the student to improve in the area where he did the mistake
Each evaluation object MUST include a "verdict" field:
- Set to "correct" if full marks were awarded.
- Set to "wrong" if anything was deducted, even partially.
This field is mandatory. Do not skip it. U MUST MENTION THE VERDICT FIELD

{{
  "evaluations": [
    {{
      "question_number": "2(i)(b)",
      "section": "A",
      "question": (question from question given)
      "type": (question type given)
      "verdict": "wrong",
      "marks_awarded": 0,
      "mistake": "Class I Lever was wrong identification because of wrong concept",
      "correct_answer": [
        "Class II lever"
      ],
      "mistake_type": "conceptual"
      "feedback": "This is a class II lever because (your explanation). Student needs to focus on these areas (proceed to give areas of weakness)",
    }},
    {{
      "question_number": "2(ii)(a)",
      "section": "A",
      "question": (question from question given)
      "type": (question type given)      
      "marks_awarded": 1,
      "verdict": "correct",
      "mistake": [],
      "correct_answer": (answer from answer key),
      "mistake_type": []
      "feedback": []
    }}
  ]
}}

⚠️ Notes:
- Always use valid JSON syntax.
- If the answer is perfect, return an empty array for `missing_or_wrong` and still include a short feedback.
- Never add markdown, headings, or extra text outside the JSON.

---

{question_blocks}

⚠️ IMPORTANT: Do not hallucinate or guess. Evaluate strictly based on the answer key.
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
