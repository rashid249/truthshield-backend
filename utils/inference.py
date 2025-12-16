import requests
import base64
import os

HF_API_KEY = os.getenv("HF_API_KEY", None)
HF_API_URL = "https://router.huggingface.co/inference"


def make_headers():
    headers = {"Content-Type": "application/json"}
    if HF_API_KEY:
        headers["Authorization"] = f"Bearer {HF_API_KEY}"
    return headers


# -----------------------------
# TEXT INFERENCE (Correct Router Format)
# -----------------------------
def hf_text_inference(model_id: str, text: str):
    try:
        payload = {
            "model": model_id,
            "inputs": text,          # ✔ FIXED (must be "inputs")
        }

        response = requests.post(
            HF_API_URL,
            headers=make_headers(),
            json=payload,
            timeout=30
        )

        return response.json()
    except Exception as e:
        return {"error": str(e)}


# -----------------------------
# IMAGE INFERENCE (Correct Router Format)
# -----------------------------
def hf_image_inference(model_id: str, image_bytes: bytes):
    try:
        img_b64 = base64.b64encode(image_bytes).decode("utf-8")

        payload = {
            "model": model_id,
            "inputs": img_b64,       # ✔ FIXED (must be "inputs")
        }

        response = requests.post(
            HF_API_URL,
            headers=make_headers(),
            json=payload,
            timeout=30
        )

        return response.json()
    except Exception as e:
        return {"error": str(e)}
