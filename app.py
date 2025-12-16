from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import os
import requests

app = FastAPI(title="TruthShield Backend")

# -----------------------------
# CORS
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Allow Chrome extension & any frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# HuggingFace Router API
# -----------------------------
HF_API_URL = "https://router.huggingface.co/inference"
HF_API_KEY = os.getenv("HF_API_KEY")   # Optional


def make_headers():
    headers = {"Content-Type": "application/json"}
    if HF_API_KEY:
        headers["Authorization"] = f"Bearer {HF_API_KEY}"
    return headers


# -----------------------------
# TEXT INFERENCE
# -----------------------------
def hf_text_inference(model: str, text: str):
    payload = {
        "model": model,
        "text": text
    }

    response = requests.post(
        HF_API_URL,
        headers=make_headers(),
        json=payload
    )

    return response.json()


# -----------------------------
# IMAGE INFERENCE
# -----------------------------
def hf_image_inference(model: str, image_bytes: bytes):
    img_b64 = base64.b64encode(image_bytes).decode()

    payload = {
        "model": model,
        "image": img_b64
    }

    response = requests.post(
        HF_API_URL,
        headers=make_headers(),
        json=payload
    )

    return response.json()


# -----------------------------
# TEXT ROUTE
# -----------------------------
class TextRequest(BaseModel):
    text: str


@app.post("/analyze/text")
def analyze_text(req: TextRequest):
    text = req.text

    sentiment = hf_text_inference("cardiffnlp/twitter-roberta-base-sentiment-latest", text)
    hate = hf_text_inference("Hate-speech-CNERG/dehatebert-mono", text)
    toxicity = hf_text_inference("facebook/roberta-hate-speech-dynabench-r4-target", text)

    return {
        "sentiment": sentiment,
        "hate_speech": hate,
        "toxicity": toxicity
    }


# -----------------------------
# IMAGE ROUTE
# -----------------------------
@app.post("/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    image_bytes = await file.read()

    nsfw = hf_image_inference("falconsai/nsfw_image_detection", image_bytes)
    objects = hf_image_inference("facebook/detr-resnet-50", image_bytes)

    return {
        "nsfw": nsfw,
        "objects": objects
    }


# -----------------------------
# ROOT
# -----------------------------
@app.get("/")
def root():
    return {"status": "TruthShield HF Router running"}
