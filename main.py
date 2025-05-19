from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import json

# Import necessary functions from planner.py
from planner import create_planner_prompt, ask_groq_api
from grader import evaluate_answer, evaluate_answer_batch
from questions import get_questions
from doubtsolver import solve_doubt

# Initialize FastAPI app
app = FastAPI()
print("App starts")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class GradeRequestItem(BaseModel):
    section: str
    question_number: str
    student_answer: str

class GradeRequest(BaseModel):
    questions: List[GradeRequestItem]

class DoubtRequest(BaseModel):
    prompt: str
    important: bool = False
    context: List[str] = []  # optional list of previous messages

class PlannerRequest(BaseModel):
    subjects: List[str]  # Subjects the student wants to focus on
    chapters: List[str]  # Specific chapters for each subject
    study_goals: str  # Overall study goals, e.g., "Improve problem-solving skills"
    strengths: List[str]  # Areas where the student is strong
    weaknesses: List[str]  # Areas of improvement
    time_available: int
    target: List[int]  # Total hours available for study each day
    days_until_target: int
    days_per_week: List[str]
    start_date: List[int]  # Days remaining until the exam

# Route: Get Questions
@app.get("/questions")
def fetch_questions():
    return get_questions()

# Route: Grade single answer
@app.post("/grade/unit")
def grade_answer(req: GradeRequestItem):
    result = evaluate_answer(req.question_number, req.student_answer, req.section)
    return {"evaluation": result}

# Route: Grade batch
@app.post("/grade_batch")
def grade_batch(req: GradeRequest):
    result_str = evaluate_answer_batch([item.model_dump() for item in req.questions])
    try:
        result_json = json.loads(result_str)
        return JSONResponse(content=result_json)
    except json.JSONDecodeError as e:
        return {
            "error": "Invalid JSON returned by LLM",
            "details": str(e),
            "raw_response": result_str,
        }

# Route: Solve doubt
@app.post("/solve_doubt")
def solve_doubt_endpoint(req: DoubtRequest):
    answer = solve_doubt(req.prompt, req.important, req.context)
    return {"response": answer}

# New Route: Generate Study Planner

@app.post("/generate_planner")
def generate_planner(req: PlannerRequest):
    prompt = create_planner_prompt(req)
    raw_response = ask_groq_api(prompt, "llama3-8b-8192")
    
    try:
        # Ensure it's valid JSON
        parsed = json.loads(raw_response)
        return JSONResponse(content=parsed)
    except json.JSONDecodeError as e:
        return {
            "error": "Invalid JSON returned by LLM",
            "details": str(e),
            "raw_response": raw_response
        }

