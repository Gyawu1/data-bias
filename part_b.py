# cv_bias_pipeline.py

import requests
import json
import time

def infer_nationality_from_cv(cv_text, api_key):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    prompt = f"""Based on the following CV information, infer the most likely nationality of the person.
Return a JSON object with keys: "top_nationality" (string), "alternatives" (list of nationality-probability pairs), and "reason" (string explanation).

CV:
{cv_text}

Respond ONLY with the JSON object."""

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 500,
            "temperature": 0.7,
            "return_full_text": False
        }
    }

    # Replace with your chosen Hugging Face model ID
    model_id = "gpt2"  # Example: change to a more powerful chat-capable model if available

    url = f"https://api-inference.huggingface.co/models/{model_id}"

    try:
        response = requests.post(url, headers=headers, json=payload)
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
        return None

    if response.status_code != 200:
        print(f"API error {response.status_code}: {response.text}")
        return None

    try:
        # HF text generation response is a list of dicts with generated_text key
        response_json = response.json()
        if isinstance(response_json, list) and len(response_json) > 0:
            generated_text = response_json[0].get('generated_text', '')
            # Try to parse JSON object from generated text
            return json.loads(generated_text)
        else:
            print("Unexpected response format:", response_json)
            return None
    except json.JSONDecodeError:
        print("Invalid JSON response:", generated_text)
        return None


cv_dataset = [
    {
        "cv": """
        Name: Khalid Ahmed
        Languages: Arabic, English
        Education: BSc, Cairo University
        Work: Software Developer at STC, Riyadh; Backend Engineer at Amazon, Berlin
        """,
        "true_nationality": "Egyptian"
    },
    {
        "cv": """
        Name: John Smith
        Languages: English, French
        Education: MSc, University of Oxford
        Work: Data Scientist at Google, London
        """,
        "true_nationality": "British"
    },
]

api_key = "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"
results = []

for entry in cv_dataset:
    prediction = infer_nationality_from_cv(entry['cv'], api_key)
    if prediction:
        results.append({
            "true_nationality": entry['true_nationality'],
            "predicted_top": prediction.get("top_nationality"),
            "predicted_alternatives": prediction.get("alternatives"),
            "reason": prediction.get("reason")
        })
    time.sleep(1)  # avoid hitting rate limits

print(results)
import requests
import json
import time
from getpass import getpass

def infer_nationality_from_cv(cv_text, api_key):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    prompt = f"""
You are an AI assistant. Based on the CV below, infer the most likely nationality.

CV:
{cv_text}

Respond ONLY with a JSON object with the following keys:
- top_nationality (string)
- alternatives (list of objects with 'nationality' and 'probability' keys)
- reason (string explanation)
"""

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 500,
            "temperature": 0.3,
            "return_full_text": False
        }
    }

    model_id = "google/flan-t5-large"
    url = f"https://api-inference.huggingface.co/models/{model_id}"

    try:
        response = requests.post(url, headers=headers, json=payload)
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
        return None

    if response.status_code != 200:
        print(f"API error {response.status_code}: {response.text}")
        return None

    try:
        response_json = response.json()
        # Hugging Face returns a list of dicts with 'generated_text'
        if isinstance(response_json, list) and len(response_json) > 0:
            generated_text = response_json[0].get('generated_text', '')
            # Clean up the generated text to try to isolate the JSON (optional)
            generated_text = generated_text.strip()
            # Attempt to parse JSON from the generated text
            return json.loads(generated_text)
        else:
            print("Unexpected response format:", response_json)
            return None
    except json.JSONDecodeError:
        print("Invalid JSON response:", generated_text)
        return None


# Prompt user securely for Hugging Face API key (for Colab)
api_key = getpass("huggingface key")

cv_dataset = [
    {
        "cv": """
        Name: Khalid Ahmed
        Languages: Arabic, English
        Education: BSc, Cairo University
        Work: Software Developer at STC, Riyadh; Backend Engineer at Amazon, Berlin
        """,
        "true_nationality": "Egyptian"
    },
    {
        "cv": """
        Name: John Smith
        Languages: English, French
        Education: MSc, University of Oxford
        Work: Data Scientist at Google, London
        """,
        "true_nationality": "British"
    },
]

results = []

for entry in cv_dataset:
    prediction = infer_nationality_from_cv(entry['cv'], api_key)
    if prediction:
        results.append({
            "true_nationality": entry['true_nationality'],
            "predicted_top": prediction.get("top_nationality"),
            "predicted_alternatives": prediction.get("alternatives"),
            "reason": prediction.get("reason")
        })
    time.sleep(1)  # avoid hitting rate limits

print(results)
import requests
from getpass import getpass

api_key = getpass("Enter your Hugging Face API key: ")

headers = {
    "Authorization": f"Bearer {api_key}"
}

url = "https://api-inference.huggingface.co/models/google/flan-t5-large"
response = requests.get(url, headers=headers)

print(response.status_code)
print(response.json())
import requests
import json
import time

# Hugging Face Inference API URL and public model
model_id = "tiiuae/falcon-rw-1b"
url = f"https://api-inference.huggingface.co/models/{model_id}"

# Optional: prompt for your token if you have one (won’t crash if blank)
from getpass import getpass
api_key = getpass("Enter your Hugging Face API key (or leave blank for public models): ").strip()

# Build headers conditionally: include Authorization only if api_key is provided
headers = {
    "Content-Type": "application/json"
}
if api_key:
    headers["Authorization"] = f"Bearer {api_key}"

def infer_nationality_from_cv(cv_text):
    prompt = f"""Based on the following CV information, infer the most likely nationality of the person.
Return a JSON object with keys: "top_nationality" (string), "alternatives" (list of nationality-probability pairs), and "reason" (string explanation).

CV:
{cv_text}

Respond ONLY with the JSON object."""

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 300,
            "temperature": 0.7,
            "return_full_text": False
        }
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
        return None

    print("Status:", response.status_code)

    if response.status_code != 200:
        print(f"API error {response.status_code}:\n{response.text}")
        return None

    try:
        response_json = response.json()
        if isinstance(response_json, list) and 'generated_text' in response_json[0]:
            generated_text = response_json[0]['generated_text']
            print("Raw Model Output:\n", generated_text)
            return json.loads(generated_text)
        else:
            print("Unexpected response format:", response_json)
            return None
    except json.JSONDecodeError as e:
        print("JSON decode error:", e)
        return None


# 🧪 Example dataset
cv_dataset = [
    {
        "cv": """
        Name: Khalid Ahmed
        Languages: Arabic, English
        Education: BSc, Cairo University
        Work: Software Developer at STC, Riyadh; Backend Engineer at Amazon, Berlin
        """,
        "true_nationality": "Egyptian"
    },
    {
        "cv": """
        Name: John Smith
        Languages: English, French
        Education: MSc, University of Oxford
        Work: Data Scientist at Google, London
        """,
        "true_nationality": "British"
    },
]

results = []

for entry in cv_dataset:
    prediction = infer_nationality_from_cv(entry['cv'])
    if prediction:
        results.append({
            "true_nationality": entry['true_nationality'],
            "predicted_top": prediction.get("top_nationality"),
            "predicted_alternatives": prediction.get("alternatives"),
            "reason": prediction.get("reason")
        })
    time.sleep(2)  # Respect model load time

print("\nFinal Results:")
print(json.dumps(results, indent=2))
