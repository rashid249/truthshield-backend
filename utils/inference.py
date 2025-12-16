# backend/utils/inference.py

import requests
import base64

HF_API_URL = "https://api-inference.huggingface.co/models"
HF_API_KEY = "your_hf_api_key_here"  # optional if models are public


def hf_text_inference(model: str, text: str):
    """
    HuggingFace API inference for text models
    """
    url = f"{HF_API_URL}/{model}"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"} if HF_API_KEY else {}
    response = requests.post(url, headers=headers, json={"inputs": text})

    try:
        return response.json()
    except:
        return {"error": "Failed to decode HF text response"}


def hf_image_inference(model: str, image_bytes: bytes):
    """
    HuggingFace API inference for image models
    """
    url = f"{HF_API_URL}/{model}"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"} if HF_API_KEY else {}
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "inputs": image_base64,
        "options": {"wait_for_model": True}
    }

    response = requests.post(url, headers=headers, json=payload)

    try:
        return response.json()
    except:
        return {"error": "Failed to decode HF image response"}
