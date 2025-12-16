# backend/utils/inference.py

import requests
import base64
import os

# HF API Key is OPTIONAL!
HF_API_KEY = os.getenv("HF_API_KEY", None)

# New HuggingFace Router endpoint (NOT api-inference.huggingface.co)
HF_API_URL = "https://router.huggingface.co/inference"


def _make_headers():
    """
    HF API Key is optional.
    If no key is provided, do not send authorization headers.
    """
    headers = {"Content-Type": "application/json"}
    if HF_API_KEY:
        headers["Authorization"] = f"Bearer {HF_API_KEY}"
    return headers


def hf_text_inference(model: str, text: str):
    """
    Unified text inference using new HF router.
    Works with OR without HF_API_KEY.
    """
    url = f"{HF_API_URL}/{model}"

    response = requests.post(
        url,
        headers=_make_headers(),
        json={"inputs": text, "options": {"wait_for_model": True}}
    )

    try:
        return response.json()
    except:
        return {"error": "Failed to parse HF text router response"}


def hf_image_inference(model: str, image_bytes: bytes):
    """
    Unified image inference using HF router.
    Works with OR without HF_API_KEY.
    """
    url = f"{HF_API_URL}/{model}"

    img_base64 = base64.b64encode(image_bytes).decode("utf-8")

    response = requests.post(
        url,
        headers=_make_headers(),
        json={"inputs": img_base64, "options": {"wait_for_model": True}}
    )

    try:
        return response.json()
    except:
        return {"error": "Failed to parse HF image router response"}
