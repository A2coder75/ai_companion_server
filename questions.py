import requests
from typing import Dict

def get_questions(filename: str) -> Dict:
    """
    Fetch fields.json and return fields + PDF URL.
    """
    try:
        # Construct direct URLs
        base_url = "https://huggingface.co/datasets/A2coder75/QnA_All/resolve/main"
        json_url = f"{base_url}/{filename}/fields.json"
        pdf_url = f"{base_url}/{filename}/qpaper.pdf"

        # Fetch the JSON
        resp = requests.get(json_url)
        resp.raise_for_status()  # raise exception if 404 or other errors

        fields_data = resp.json()  # parse JSON directly
        if not isinstance(fields_data, list):
            return {"error": "fields.json is invalid: not a list"}

        return {
            "fields": fields_data,
            "pdf_url": pdf_url
        }

    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to fetch questions: {str(e)}"}
    except ValueError as e:
        return {"error": f"Invalid JSON in fields.json: {str(e)}"}
