from huggingface_hub import hf_hub_download
from typing import Dict
import json

def get_questions(filename: str) -> Dict:
    try:
        # Download fields.json to a local temp path
        fields_path = hf_hub_download(
            repo_id="A2coder75/QnA_All",
            repo_type="dataset",
            filename=f"{filename}/fields.json"
        )

        # Load JSON from the downloaded file
        with open(fields_path, "r", encoding="utf-8") as f:
            fields_data = json.load(f)

        # Construct direct PDF URL
        pdf_url = f"https://huggingface.co/datasets/A2coder75/QnA_All/resolve/main/{filename}/qpaper.pdf"

        return {"fields": fields_data, "pdf_url": pdf_url}

    except Exception as e:
        return {"error": f"Failed to fetch questions: {str(e)}"}
