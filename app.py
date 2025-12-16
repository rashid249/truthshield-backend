from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests, base64, os

# -----------------------
# CONFIG
# -----------------------
HF_API_KEY = os.getenv("HF_API_KEY")
HF_API_URL = "https://router.huggingface.co/inference"

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# HELPERS
# -----------------------
def hf_request(payload):
    headers = {"Content-Type": "application/json"}
    if HF_API_KEY:
        headers["Authorization"] = f"Bearer {HF_API_KEY}"

    try:
        r = requests.post(HF_API_URL, headers=headers, json=payload, timeout=25)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

# -----------------------
# TEXT INFERENCE
# -----------------------
def analyze_text_models(text):
    models = {
        "sentiment": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "hate": "Hate-speech-CNERG/dehatebert-mono",
        "toxicity": "facebook/roberta-hate-speech-dynabench-r4-target",
        "bias": "mmathis/bias-detection-roberta",
        "propaganda": "iviarcio/propaganda-classification"
    }

    result = {}
    for key, model in models.items():
        result[key] = hf_request({"model": model, "text": text})

    return result

def compute_trust_score(res):
    try:
        hate = float(res["hate"][0]["score"])
        tox = float(res["toxicity"][0]["score"])
        bias = float(res["bias"][0]["score"])

        # propaganda avg score
        propaganda = (
            sum([x["score"] for x in res["propaganda"]]) /
            len(res["propaganda"])
            if isinstance(res["propaganda"], list) else 0
        )

        trust = 1 - (0.3*hate + 0.3*tox + 0.2*bias + 0.2*propaganda)
        return max(0, min(1, trust))
    except:
        return 0.5

# -----------------------
# REQUEST BODY
# -----------------------
class TextRequest(BaseModel):
    text: str


# -----------------------
# ENDPOINTS
# -----------------------
@app.post("/analyze/text")
def analyze_text(req: TextRequest):
    text = req.text
    res = analyze_text_models(text)
    trust_score = compute_trust_score(res)

    return {
        "trust_score": trust_score,
        "details": res
    }


@app.post("/analyze/selected")
def analyze_selected(req: TextRequest):
    return analyze_text(req)


@app.post("/analyze/image")
async def analyze_image(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img_b64 = base64.b64encode(img_bytes).decode()

    nsfw = hf_request({"model": "falconsai/nsfw_image_detection", "image": img_b64})
    objects = hf_request({"model": "facebook/detr-resnet-50", "image": img_b64})

    return {
        "nsfw": nsfw,
        "objects": objects
    }


@app.get("/")
def root():
    return {"status": "TruthShield Backend Running"}
