from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import requests
import os

app = FastAPI()

# -----------------------------
# CORS (Important for Chrome/Edge Extensions)
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # you can restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Environment Variables (Railway)
# -----------------------------
HF_API_KEY = os.getenv("HF_API_KEY")   # Set this in Railway Variables

if not HF_API_KEY:
    raise ValueError("❌ HF_API_KEY is missing — Add it in Railway environment variables")

HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}


# -----------------------------
# Text Inference Helper
# -----------------------------
def hf_text_inference(model_id: str, text: str):
    try:
        res = requests.post(
            f"https://api-inference.huggingface.co/models/{model_id}",
            headers=HEADERS,
            json={"inputs": text},
        )
        return res.json()
    except Exception as e:
        return {"error": str(e)}


# -----------------------------
# Image Inference Helper (Correct format)
# -----------------------------
def hf_image_inference(model_id: str, image_bytes: bytes):
    try:
        res = requests.post(
            f"https://api-inference.huggingface.co/models/{model_id}",
            headers=HEADERS,
            files={"file": image_bytes},   # correct format for image models
        )
        return res.json()
    except Exception as e:
        return {"error": str(e)}


# -----------------------------
# Text Schema
# -----------------------------
class TextRequest(BaseModel):
    text: str


# -----------------------------
# ROUTES
# -----------------------------

@app.get("/")
def root():
    return {"status": "TruthShield backend (HuggingFace version) running"}


# -----------------------------
# MATCHING FRONTEND ROUTE
# POST /analyze/text
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
# MATCHING FRONTEND ROUTE
# POST /analyze/image
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
