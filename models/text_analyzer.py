# backend/models/text_analyzer.py

from fastapi import APIRouter
from backend.utils.schema import TextRequest
from backend.utils.inference import hf_text_inference

router = APIRouter(prefix="/analyze")


@router.post("/text")
def analyze_text(data: TextRequest):
    text = data.text

    sentiment = hf_text_inference(
        "cardiffnlp/twitter-roberta-base-sentiment-latest",
        text
    )

    hate = hf_text_inference(
        "Hate-speech-CNERG/dehatebert-mono",
        text
    )

    toxicity = hf_text_inference(
        "facebook/roberta-hate-speech-dynabench-r4-target",
        text
    )

    return {
        "sentiment": sentiment,
        "hate_speech": hate,
        "toxicity": toxicity
    }
