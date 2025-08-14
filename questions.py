from huggingface_hub import hf_hub_download
import json

def get_questions(filename: str):
    try:
        fields_path = hf_hub_download(
            repo_id="A2coder75/QnA_All",
            repo_type="dataset",
            filename=f"{filename}/fields.json"
        )

        # Debug: read raw content
        with open(fields_path, "r", encoding="utf-8") as f:
            content = f.read()
        print("Raw content preview:", content[:500], flush=True)  # first 500 chars

        fields_data = json.loads(content)  # parse JSON
        pdf_url = f"https://huggingface.co/datasets/A2coder75/QnA_All/resolve/main/{filename}/qpaper.pdf"

        return {"fields": fields_data, "pdf_url": pdf_url}

    except Exception as e:
        print("Error in get_questions:", str(e), flush=True)
        return {"error": f"Failed to fetch questions: {str(e)}"}
