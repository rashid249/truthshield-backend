# backend/utils/inference.py

import requests
import base64
import os

HF_API_KEY = os.getenv("HF_API_KEY")
HF_API_URL = "https://router.huggingface.co/inference"


def hf_text_inference(model: str, text: str):
    """
    New HuggingFace Router inference for text models
    """
    url = f"{HF_API_URL}/{model}"
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "inputs": text,
        "options": {"wait_for_model": True}
    }

    response = requests.post(url, headers=headers, json=payload)
    try:
        return response.json()
    except:
        return {"error": "Failed to parse HF router response"}


def hf_image_inference(model: str, image_bytes: bytes):
    """
    New HuggingFace Router inference for image models
    """
    url = f"{HF_API_URL}/{model}"
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json"
    }

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "inputs": image_base64,
        "options": {"wait_for_model": True}
    }

    response = requests.post(url, headers=headers, json=payload)
    try:
        return response.json()
    except:
        return {"error": "Failed to parse HF router response"}
