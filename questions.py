import json
import os
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

def get_questions(filename: str):
    """
    Fetch questions JSON file from Hugging Face repo QnA_All using the given filename.
    """
    try:
        file_path = hf_hub_download(
            repo_id="A2coder75/QnA_All",
            repo_type="dataset",
            filename=filename,
            token=HF_TOKEN
        )
        with open(file_path, "r", encoding="utf-8") as f:
            questions_data = json.load(f)
        return questions_data
    except Exception as e:
        return {"error": f"Failed to fetch questions: {str(e)}"}
