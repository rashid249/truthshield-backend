# backend/models/image_analyzer.py

from fastapi import UploadFile, File
from backend.app import app
from backend.utils.inference import hf_image_inference

@app.post("/analyze_image")
async def analyze_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()

        # -----------------------------
        # NSFW MODEL
        # -----------------------------
        nsfw = hf_image_inference(
            "falconsai/nsfw_image_detection",
            image_bytes
        )

        # -----------------------------
        # OBJECT DETECTION (DETR)
        # -----------------------------
        detr = hf_image_inference(
            "facebook/detr-resnet-50",
            image_bytes
        )

        # -----------------------------
        # NSFW SCORE
        # -----------------------------
        nsfw_score = 0.0
        try:
            porn_item = next(
                (x for x in nsfw if x["label"] in ["porn", "sexual", "hentai"]),
                None
            )
            if porn_item:
                nsfw_score = float(porn_item["score"])
        except:
            nsfw_score = 0.0

        # -----------------------------
        # DETR OBJECT CONFIDENCES
        # -----------------------------
        detr_objects = []
        detr_conf = []

        try:
            preds = detr["outputs"] if isinstance(detr, dict) and "outputs" in detr else detr

            for obj in preds:
                if "score" in obj:
                    label = obj.get("label", "object")
                    score = float(obj["score"])

                    detr_objects.append((label, score))
                    detr_conf.append(score)
        except:
            pass

        top_objects = sorted(detr_objects, key=lambda x: -x[1])[:3]
        details = (
            ", ".join([f"{lbl}({conf:.2f})" for lbl, conf in top_objects])
            if top_objects else "No objects detected"
        )

        avg_object_conf = (
            sum(detr_conf) / len(detr_conf) if len(detr_conf) > 0 else 0
        )

        # -----------------------------
        # AI-LIKENESS SCORE
        # -----------------------------
        ai_score = (
            0.60 * nsfw_score +
            0.40 * (1 - avg_object_conf)
        )
        ai_score = max(0, min(1, ai_score))

        if ai_score < 0.25:
            label = "likely_real"
        elif ai_score < 0.55:
            label = "uncertain"
        else:
            label = "possibly_ai_or_edited"

        # -----------------------------
        # RETURN
        # -----------------------------
        return {
            "label": label,
            "ai_score": ai_score,
            "details": details,
            "nsfw_score": nsfw_score,
            "objects": detr
        }

    except Exception as e:
        return {
            "label": "error_loading_image",
            "ai_score": 0.0,
            "details": str(e)
        }
