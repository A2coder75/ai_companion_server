from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List
import json

# Import necessary functions
from planner import create_planner_prompt, ask_groq_api
from grader import evaluate_answer_batch
from questions import get_questions
from doubtsolver import solve_doubt

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

# ============================
# Pydantic Models
# ============================

class DoubtRequest(BaseModel):
    prompt: str
    important: bool = False
    context: List[str] = []

class PlannerRequest(BaseModel):
    subjects: List[str]
    chapters: List[str]
    study_goals: str
    strengths: List[str]
    weaknesses: List[str]
    time_available: int
    target: List[int]
    days_until_target: int
    days_per_week: List[str]
    start_date: List[int]

class QuestionsRequest(BaseModel):
    filename: str

class AnswerItem(BaseModel):
    question_id: str
    answer: str

# ============================
# Routes
# ============================

# Route: Get Questions
@app.post("/questions")
def fetch_questions(req: QuestionsRequest):
    result = get_questions(req.filename)
    if "error" in result:
        return JSONResponse(content=result)
    return JSONResponse(content={
        "fields": result.get("fields", []),
        "pdf_url": result.get("pdf_url", "")
    })


# Route: Download PDF
@app.get("/download_pdf")
def download_pdf(path: str):
    return FileResponse(path, media_type="application/pdf", filename="qpaper.pdf")

# Route: Grade batch
@app.post("/grade_batch")
def grade_batch(req: List[AnswerItem]):
    # Convert Pydantic objects to dict for processing
    batch_data = [item.dict() for item in req]
    result_str = evaluate_answer_batch(batch_data)
    try:
        result_json = json.loads(result_str)
        return JSONResponse(content=result_json)
    except json.JSONDecodeError as e:
        return JSONResponse(content={
            "error": "Invalid JSON returned by LLM",
            "details": str(e),
            "raw_response": result_str
        })

# Route: Solve doubt
@app.post("/solve_doubt")
def solve_doubt_endpoint(req: DoubtRequest):
    answer = solve_doubt(req.prompt, req.important, req.context)
    return JSONResponse(content={"response": answer})

# Route: Generate Study Planner
@app.post("/generate_planner")
def generate_planner(req: PlannerRequest):
    prompt = create_planner_prompt(req)
    raw_response = ask_groq_api(prompt, "llama3-8b-8192")
    try:
        parsed = json.loads(raw_response)
        return JSONResponse(content=parsed)
    except json.JSONDecodeError as e:
        return JSONResponse(content={
            "error": "Invalid JSON returned by LLM",
            "details": str(e),
            "raw_response": raw_response
        })

# ============================
# Health check endpoint
# ============================
@app.get("/health")
def health_check():
    return JSONResponse(content={"status": "ok"})

