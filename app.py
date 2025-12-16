from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import os
import requests

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# HF Router API
HF_API_URL = "https://router.huggingface.co/inference"
HF_API_KEY = os.getenv("HF_API_KEY")

def make_headers():
    headers = {"Content-Type": "application/json"}
    if HF_API_KEY:
        headers["Authorization"] = f"Bearer {HF_API_KEY}"
    return headers


# SAFE JSON EXTRACTOR (prevents crashes)
def safe_json(response):
    try:
        return response.json()
    except:
        return {
            "error": "Non-JSON response from HF Router",
            "status_code": response.status_code,
            "text": response.text
        }


# TEXT inference (HF Router compatible)
def hf_text_inference(model: str, text: str):
    payload = {
        "model": model,
        "inputs": [text]   # <-- FIX: HF Router requires list
    }

    resp = requests.post(
        HF_API_URL,
        headers=make_headers(),
        json=payload
    )

    return safe_json(resp)


# IMAGE inference (HF Router compatible)
def hf_image_inference(model: str, image_bytes: bytes):
    img_b64 = base64.b64encode(image_bytes).decode()

    payload = {
        "model": model,
        "inputs": img_b64
    }

    resp = requests.post(
        HF_API_URL,
        headers=make_headers(),
        json=payload
    )

    return safe_json(resp)


# TEXT ROUTE
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


# IMAGE ROUTE
@app.post("/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    image_bytes = await file.read()

    nsfw = hf_image_inference("falconsai/nsfw_image_detection", image_bytes)
    objects = hf_image_inference("facebook/detr-resnet-50", image_bytes)

    return {"nsfw": nsfw, "objects": objects}


@app.get("/")
def root():
    return {"status": "TruthShield HF Router running"}
