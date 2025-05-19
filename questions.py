from huggingface_hub import hf_hub_download
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import os
import json

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Download questions from Hugging Face
questions_file_path = hf_hub_download(
    repo_id="A2coder75/icse_board_paper_2024_physics",
    repo_type="dataset",
    filename="IcseXPhysicsPaper2024.json",
    token=HF_TOKEN,
    force_download=True
)

def get_questions():
    with open(questions_file_path, "r", encoding="utf-8") as f:
        return JSONResponse(content=json.load(f))
