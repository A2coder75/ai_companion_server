from huggingface_hub import hf_hub_download
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import os
import json

from dotenv import dotenv_values
from io import StringIO

# Parse multi-line TOKENS env secret
tokens_raw = os.getenv("TOKENS")
if tokens_raw:
    env_vars = dotenv_values(stream=StringIO(tokens_raw))
    HF_TOKEN = env_vars.get("HF_TOKEN")
else:
    raise EnvironmentError("TOKENS secret not found in environment.")

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
