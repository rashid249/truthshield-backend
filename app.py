from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import requests
import base64
import os

app = FastAPI()

# -----------------------------
# CORS (Chrome Extension Support)
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# HuggingFace Router Endpoint
# -----------------------------
HF_API_KEY = os.getenv("HF_API_KEY", None)   # optional
HF_ROUTER_URL = "https://router.huggingface.co/inference"


def make_headers():
    """HF API key is optional."""
    headers = {"Content-Type": "application/json"}
    if HF_API_KEY:
        headers["Authorization"] = f"Bearer {HF_API_KEY}"
    return headers


# -----------------------------
# TEXT INFERENCE (Correct HF Router Format)
# -----------------------------
def hf_text_inference(model_id: str, text: str):
    """
    HuggingFace Router expects:
    
    {
       "model": "model-name",
       "inputs": "some text"
    }
    """
    try:
        payload = {
            "model": model_id,
            "inputs": text
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
    """
    HF Router expects base64 string as inputs, not nested JSON.
    
    {
       "model": "model-name",
       "inputs": "<base64 string>"
    }
    """
    try:
        img_b64 = base64.b64encode(image_bytes).decode("utf-8")

        payload = {
            "model": model_id,
            "inputs": img_b64
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
# Request Schema
# -----------------------------
class TextRequest(BaseModel):
    text: str


# -----------------------------
# Root Endpoint
# -----------------------------
@app.get("/")
def root():
    return {
        "status": "TruthShield backend running (HF Router enabled)",
        "hf_api_key_loaded": HF_API_KEY is not None
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
