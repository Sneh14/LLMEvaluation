import json
from pathlib import Path

from langchain import requests

def load_test_data(filename):
    test_data_path = Path(__file__).parent.absolute() / 'testData' / filename
    with open(test_data_path) as f:
        return json.load(f)

def get_llm_response(test_data):
    responsedict = requests.post(
        url="https://rahulshettyacademy.com/rag-llm/ask",
        json={
            "question": test_data["question"],
            "chat_history": [
            ]
        }
    ).json()

    return responsedict
