import json
import os
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

def get_questions(filename: str):
    """
    Fetch fields.json from Hugging Face repo QnA_All 
    and return with the direct qpaper.pdf URL.
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

        # Construct direct PDF URL (public link)
        pdf_url = (
            f"https://huggingface.co/datasets/A2coder75/QnA_All/resolve/main/{filename}/qpaper.pdf"
        )

        return {
            "fields": fields_data,
            "pdf_url": pdf_url
        }

    except Exception as e:
        return {"error": f"Failed to fetch questions: {str(e)}"}
