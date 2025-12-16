@app.post("/analyze_image")
async def analyze_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()

        # NSFW MODEL
        nsfw = hf_image_inference(
            "falconsai/nsfw_image_detection", image_bytes
        )

        # DETR OBJECT DETECTION
        detr = hf_image_inference(
            "facebook/detr-resnet-50", image_bytes
        )

        # -----------------------------
        # Extract NSFW score
        # -----------------------------
        nsfw_score = 0.0
        try:
            # The NSFW model returns predictions like:
            # [{"label": "neutral", "score": 0.95}, {"label": "porn", "score": 0.02}, ...]
            porn_item = next((x for x in nsfw if x["label"] in ["porn", "sexual", "hentai"]), None)
            if porn_item:
                nsfw_score = float(porn_item["score"])
        except:
            nsfw_score = 0.0

        # -----------------------------
        # Extract DETR confidence
        # -----------------------------
        detr_objects = []
        detr_confidences = []

        if isinstance(detr, dict) and "outputs" in detr:
            preds = detr["outputs"]
        else:
            preds = detr  # fallback

        try:
            for obj in preds:
                if "score" in obj and obj["score"] is not None:
                    conf = float(obj["score"])
                    label = obj.get("label", "object")
                    detr_objects.append((label, conf))
                    detr_confidences.append(conf)
        except:
            pass

        # Top 3 objects for details
        top_objects = sorted(detr_objects, key=lambda x: -x[1])[:3]
        details = ", ".join([f"{lbl}({conf:.2f})" for lbl, conf in top_objects]) if top_objects else "No objects detected"

        # Average DETR confidence = "realness" of natural photography
        avg_object_conf = sum(detr_confidences) / len(detr_confidences) if detr_confidences else 0

        # -----------------------------
        # AI-LIKENESS SCORE (final)
        # -----------------------------
        # Formula:
        #   More NSFW → more likely AI / manipulated
        #   Lower DETR confidence → less natural → more likely AI
        #
        # ai_score = probability image is AI-generated (0 to 1)

        ai_score = (
            0.60 * nsfw_score +               # NSFW = strong AI indicator
            0.40 * (1 - avg_object_conf)      # low object confidence = AI-like image
        )

        ai_score = max(0, min(1, ai_score))

        # -----------------------------
        # LABEL DECISION
        # -----------------------------
        if ai_score < 0.25:
            label = "likely_real"
        elif ai_score < 0.55:
            label = "uncertain"
        else:
            label = "possibly_ai_or_edited"

        return {
            "label": label,
            "ai_score": ai_score,
            "details": details,
            "nsfw_score": nsfw_score,
            "objects": detr  # full raw model output
        }

    except Exception as e:
        return {
            "label": "error_loading_image",
            "ai_score": 0.0,
            "details": str(e)
        }
