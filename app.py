from fastapi import FastAPI, UploadFile, File
import requests

app = FastAPI()

# -----------------------------
# HuggingFace API Config
# -----------------------------

HF_API_KEY = ""  # Optional: put your HF API Key here
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"} if HF_API_KEY else {}

TEXT_MODEL_URL = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment-latest"
IMAGE_MODEL_URL = "https://api-inference.huggingface.co/models/google/vit-base-patch16-224"


# -----------------------------
# Root check
# -----------------------------

@app.get("/")
def root():
    return {"status": "TruthShield backend running (HF Powered)"}


# -----------------------------
# TEXT ANALYSIS
# -----------------------------

@app.post("/analyze_text")
async def analyze_text(payload: dict):
    text = payload.get("text", "")

    if not text:
        return {"error": "Text is required"}

    response = requests.post(TEXT_MODEL_URL, headers=HEADERS, json={"inputs": text})
    
    return {"input": text, "result": response.json()}


# -----------------------------
# IMAGE ANALYSIS
# -----------------------------

@app.post("/analyze_image")
async def analyze_image(file: UploadFile = File(...)):
    image_bytes = await file.read()

    response = requests.post(IMAGE_MODEL_URL, headers=HEADERS, data=image_bytes)

    return {"filename": file.filename, "result": response.json()}
