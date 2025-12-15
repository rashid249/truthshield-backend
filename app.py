from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import requests
import base64

app = FastAPI()

# -----------------------------
# HuggingFace API Helper
# -----------------------------

HF_API_KEY = "YOUR_HF_API_KEY"   # ← Replace with your token

HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

def hf_text_inference(model_id, text):
    response = requests.post(
        f"https://api-inference.huggingface.co/models/{model_id}",
        headers=HEADERS,
        json={"inputs": text},
    )
    return response.json()

def hf_image_inference(model_id, image_bytes):
    img_base64 = base64.b64encode(image_bytes).decode("utf-8")

    response = requests.post(
        f"https://api-inference.huggingface.co/models/{model_id}",
        headers=HEADERS,
        json={"inputs": img_base64},
    )
    return response.json()

# -----------------------------
# Text Input Schema
# -----------------------------

class TextRequest(BaseModel):
    text: str

# -----------------------------
# ROUTES
# -----------------------------

@app.get("/")
def root():
    return {"status": "TruthShield backend (HuggingFace version) running"}

@app.post("/analyze_text")
def analyze_text(data: TextRequest):
    text = data.text

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


@app.post("/analyze_image")
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
