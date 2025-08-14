import json
import os
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

def get_questions(filename: str):
    """
    Fetch two files from Hugging Face repo QnA_All:
    1. filename/fields.json
    2. filename/qpaper.pdf
    """
    try:
        # Download fields.json
        fields_path = hf_hub_download(
            repo_id="A2coder75/QnA_All",
            repo_type="dataset",
            filename=f"{filename}/fields.json",
            token=HF_TOKEN
        )
        with open(fields_path, "r", encoding="utf-8") as f:
            fields_data = json.load(f)

        # Download qpaper.pdf
        pdf_path = hf_hub_download(
            repo_id="A2coder75/QnA_All",
            repo_type="dataset",
            filename=f"{filename}/qpaper.pdf",
            token=HF_TOKEN
        )

        # Return both
        return {
            "fields": fields_data,
            "pdf_path": pdf_path  # Can be sent as a file in Flask/FastAPI response
        }

    except Exception as e:
        return {"error": f"Failed to fetch files: {str(e)}"}
