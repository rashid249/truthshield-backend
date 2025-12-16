# backend/models/text_analyzer.py

from fastapi import APIRouter
from backend.app import app
from backend.utils.schema import TextRequest
from backend.utils.inference import hf_text_inference

@app.post("/analyze_text")
def analyze_text(data: TextRequest):
    text = data.text

    # ---- BASE MODELS ----
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

    # ---- NEW MODELS ----
    bias = hf_text_inference(
        "mmathis/bias-detection-roberta",
        text
    )

    propaganda = hf_text_inference(
        "iviarcio/propaganda-classification",
        text
    )

    # -----------------------------
    # SCORE EXTRACTION
    # -----------------------------
    hate_score = float(hate[0]["score"]) if isinstance(hate, list) else 0
    tox_score = float(toxicity[0]["score"]) if isinstance(toxicity, list) else 0

    bias_label = bias[0]["label"] if isinstance(bias, list) else "neutral"
    bias_score = float(bias[0]["score"]) if isinstance(bias, list) else 0

    propaganda_tags = []
    propaganda_intensity = 0

    if isinstance(propaganda, list):
        for tag in propaganda:
            if tag["score"] > 0.5:
                propaganda_tags.append(tag["label"])

        propaganda_intensity = (
            sum([tag["score"] for tag in propaganda]) / len(propaganda)
        )

    # -----------------------------
    # TRUST SCORE
    # -----------------------------
    trust = (
        1
        - (0.3 * tox_score)
        - (0.3 * hate_score)
        - (0.2 * bias_score)
        - (0.2 * propaganda_intensity)
    )
    trust = max(0, min(1, trust))

    # -----------------------------
    # EXPLANATION
    # -----------------------------
    explanation = (
        f"The text shows {bias_label} bias with score {bias_score:.2f}. "
        f"Toxicity: {tox_score:.2f}, Hate: {hate_score:.2f}. "
        f"Propaganda detected: {', '.join(propaganda_tags) if propaganda_tags else 'none'}. "
        f"Final trust score: {trust:.2f}."
    )

    # -----------------------------
    # RETURN
    # -----------------------------
    return {
        "overall_trust_score": trust,
        "bias_label": bias_label,
        "bias_score": bias_score,
        "propaganda_tags": propaganda_tags,
        "explanation": explanation,

        # Raw model outputs
        "sentiment": sentiment,
        "hate_speech": hate,
        "toxicity": toxicity,
    }

