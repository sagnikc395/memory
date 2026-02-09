# check all avilable models to try out first

import requests
import os
from dotenv import load_dotenv
import json

load_dotenv()


api_key = os.environ.get("GROQ_API_KEY")
url = "https://api.groq.com/openai/v1/models"

headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

response = requests.get(url, headers=headers)


print(json.dumps(response.json(), indent=4))
