from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import requests
import base64
import os

app = FastAPI()

# -----------------------------
# CORS (Important for Extension)
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# HuggingFace Router API
# -----------------------------
HF_API_KEY = os.getenv("HF_API_KEY", None)   # Optional

HF_ROUTER_URL = "https://router.huggingface.co/inference"


def make_headers():
    """
    Create headers dynamically.
    If no API Key → run in public mode.
    """
    headers = {"Content-Type": "application/json"}
    if HF_API_KEY:
        headers["Authorization"] = f"Bearer {HF_API_KEY}"
    return headers


# -----------------------------
# TEXT INFERENCE (Correct HF Router Format)
# -----------------------------
def hf_text_inference(model_id: str, text: str):
    try:
        payload = {
            "model": model_id,
            "inputs": {
                "text": text
            }
        }

        response = requests.post(
            HF_ROUTER_URL,
            headers=make_headers(),
            json=payload
        )

        return response.json()
    except Exception as e:
        return {"error": str(e)}


# -----------------------------
# IMAGE INFERENCE (Correct HF Router Format)
# -----------------------------
def hf_image_inference(model_id: str, image_bytes: bytes):
    try:
        img_b64 = base64.b64encode(image_bytes).decode("utf-8")

        payload = {
            "model": model_id,
            "inputs": {
                "image": img_b64
            }
        }

        response = requests.post(
            HF_ROUTER_URL,
            headers=make_headers(),
            json=payload
        )

        return response.json()

    except Exception as e:
        return {"error": str(e)}


# Schema
class TextRequest(BaseModel):
    text: str


# -----------------------------
# Root Route
# -----------------------------
@app.get("/")
def root():
    return {
        "status": "TruthShield backend running with HuggingFace Router API",
        "api_key_loaded": HF_API_KEY is not None
    }


# -----------------------------
# TEXT ANALYSIS ROUTE
# -----------------------------
@app.post("/analyze/text")
def analyze_text(request: TextRequest):
    text = request.text

    sentiment = hf_text_inference(
        "cardiffnlp/twitter-roberta-base-sentiment-latest", text
    )

    hate_speech = hf_text_inference(
        "Hate-speech-CNERG/dehatebert-mono", text
    )

    toxicity = hf_text_inference(
        "facebook/roberta-hate-speech-dynabench-r4-target", text
    )

    return {
        "sentiment": sentiment,
        "hate_speech": hate_speech,
        "toxicity": toxicity
    }


# -----------------------------
# IMAGE ANALYSIS ROUTE
# -----------------------------
@app.post("/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    image_bytes = await file.read()

    nsfw = hf_image_inference(
        "falconsai/nsfw_image_detection", image_bytes
    )

    objects = hf_image_inference(
        "facebook/detr-resnet-50", image_bytes
    )

    return {
        "nsfw": nsfw,
        "objects": objects
    }
