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

# HuggingFace Router (CORRECT ENDPOINT)
HF_API_URL = "https://router.huggingface.co/inference"
HF_API_KEY = os.getenv("HF_API_KEY")

def make_headers():
    headers = {"Content-Type": "application/json"}
    if HF_API_KEY:
        headers["Authorization"] = f"Bearer {HF_API_KEY}"
    return headers


# -----------------------------
# TEXT INFERENCE (VALID FOR HF ROUTER)
# -----------------------------
def hf_text_inference(model: str, text: str):
    payload = {
        "model": model,
        "inputs": text         # ✔ correct HF Router field
    }

    res = requests.post(HF_API_URL, headers=make_headers(), json=payload)

    if res.status_code != 200:
        return {"error": f"HF_ERROR {res.status_code}", "details": res.text}

    try:
        return res.json()
    except:
        return {"error": "INVALID_JSON", "details": res.text}


# -----------------------------
# IMAGE INFERENCE (VALID FOR HF ROUTER)
# -----------------------------
def hf_image_inference(model: str, image_bytes: bytes):
    img_b64 = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "model": model,
        "inputs": img_b64        # ✔ correct field for HF Router image input
    }

    res = requests.post(HF_API_URL, headers=make_headers(), json=payload)

    if res.status_code != 200:
        return {"error": f"HF_ERROR {res.status_code}", "details": res.text}

    try:
        return res.json()
    except:
        return {"error": "INVALID_JSON", "details": res.text}


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


@app.post("/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    image_bytes = await file.read()

    nsfw = hf_image_inference("falconsai/nsfw_image_detection", image_bytes)
    objects = hf_image_inference("facebook/detr-resnet-50", image_bytes)

    return {
        "nsfw": nsfw,
        "objects": objects
    }


@app.get("/")
def root():
    return {"status": "TruthShield HF Router BACKEND OK"}
