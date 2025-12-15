# backend/models/text_analyzer.py

from transformers import pipeline


class TextBiasAnalyzer:
    """
    Uses three HuggingFace pipelines:

    1) Bias classifier  (newsmediabias/UnBIAS-classification-bert)
    2) Sentiment analysis (CardiffNLP RoBERTa)
    3) Propaganda detection (zero-shot with BART MNLI)

    Then combines them into a trust score in [0, 1].
    """

    def __init__(self):
        # -----------------------------
        # 1. BIAS PIPELINE  (PyTorch)
        # -----------------------------
        # Binary: biased vs non-biased
        self.bias_pipe = pipeline(
            "text-classification",
            model="newsmediabias/UnBIAS-classification-bert",
            truncation=True,
            max_length=512,
        )

        # -----------------------------
        # 2. SENTIMENT PIPELINE
        # -----------------------------
        self.sentiment_pipe = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
        )

        # -----------------------------
        # 3. PROPAGANDA PIPELINE
        # -----------------------------
        self.propaganda_pipe = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
        )

        self.propaganda_labels = [
            "appeal to fear",
            "loaded language",
            "name calling",
            "flag waving",
            "bandwagon",
            "appeal to authority",
            "whataboutism",
            "straw man argument",
            "glittering generalities",
            "doubt",
        ]

    # -----------------------------
    # Helper: combine into trust score
    # -----------------------------
    def _compute_trust_score(
        self,
        bias_prob: float,
        sentiment_label: str,
        sentiment_score: float,
        num_propaganda_tags: int,
    ) -> float:
        """
        bias_prob:      0 (unbiased) → 1 (strongly biased)
        sentiment_label: NEGATIVE / NEUTRAL / POSITIVE
        sentiment_score: confidence of that label
        num_propaganda_tags: how many techniques we flagged

        Returns: trust score between 0 and 1
        """

        # 1) bias risk
        bias_risk = bias_prob

        # 2) emotional intensity
        if sentiment_label.lower().startswith("neu"):
            emotional_intensity = 0.2
        else:
            emotional_intensity = 0.3 + 0.7 * sentiment_score
            emotional_intensity = max(0.0, min(1.0, emotional_intensity))

        # 3) propaganda penalty
        max_techniques = max(1, len(self.propaganda_labels))
        # scale: a few techniques already hurt; later ones saturate
        propaganda_penalty = min(1.0, (num_propaganda_tags / max_techniques) * 2.0)

        # Softer weights: all three matter
        raw_penalty = (
            0.35 * bias_risk
            + 0.35 * emotional_intensity
            + 0.30 * propaganda_penalty
        )

        trust = 1.0 - raw_penalty

        # Neutral + no propaganda bonus
        if sentiment_label.lower().startswith("neu") and num_propaganda_tags == 0:
            # Don't go below 0.7 if text is calm & non-propagandistic
            trust = max(trust, 0.7)

        trust = float(max(0.0, min(1.0, trust)))
        return trust

    # -----------------------------
    # Main analyze() method
    # -----------------------------
    def analyze(self, text: str):
        """
        Returns a dict with:
        - bias_score: probability that text is biased (0–1)
        - bias_label: "biased" / "unbiased"
        - propaganda_tags: list of detected techniques (strings)
        - emotional_tone: simple anger/fear/hope scores
        - overall_trust_score: 0–1 (UI multiplies by 100)
        - explanation: human-readable description
        """

        short_text = text[:1000] if text else ""

        # -----------------------------
        # 1. Bias prediction
        # -----------------------------
        try:
            bias_result = self.bias_pipe(short_text)[0]
            raw_label = bias_result["label"]  # "Biased" / "Non-biased" / LABEL_0...
            score = float(bias_result["score"])

            lower = raw_label.lower()
            if "biased" in lower and "non" not in lower:
                bias_label = "biased"
                bias_prob = score
            elif "non" in lower and "bias" in lower:
                bias_label = "unbiased"
                bias_prob = 1.0 - score
            else:
                # fallback for LABEL_0 / LABEL_1
                if raw_label.endswith("1"):
                    bias_label = "biased"
                    bias_prob = score
                else:
                    bias_label = "unbiased"
                    bias_prob = 1.0 - score
        except Exception:
            bias_label = "unbiased"
            bias_prob = 0.5

        # -----------------------------
        # 2. Sentiment / emotionality
        # -----------------------------
        try:
            sent_result = self.sentiment_pipe(short_text[:512])[0]
            sentiment_label = sent_result["label"]  # NEGATIVE / NEUTRAL / POSITIVE
            sentiment_score = float(sent_result["score"])
        except Exception:
            sentiment_label = "NEUTRAL"
            sentiment_score = 0.5

        # Map to anger/fear/hope
        if sentiment_label.lower().startswith("neg"):
            anger = 0.5 + 0.4 * sentiment_score
            fear = 0.5 + 0.3 * sentiment_score
            hope = 0.1
        elif sentiment_label.lower().startswith("pos"):
            anger = 0.1
            fear = 0.2
            hope = 0.5 + 0.4 * sentiment_score
        else:
            anger = 0.2
            fear = 0.2
            hope = 0.3

        emotional_tone = {
            "anger": float(max(0.0, min(1.0, anger))),
            "fear": float(max(0.0, min(1.0, fear))),
            "hope": float(max(0.0, min(1.0, hope))),
        }

        # -----------------------------
        # 3. Propaganda detection
        # -----------------------------
        try:
            prop_result = self.propaganda_pipe(
                short_text,
                self.propaganda_labels,
                multi_label=True,
            )
            # stricter threshold to avoid false positives (e.g., Wikipedia)
            propaganda_tags = [
                label
                for label, score in zip(
                    prop_result["labels"], prop_result["scores"]
                )
                if score >= 0.60
            ]
        except Exception:
            propaganda_tags = []

        # -----------------------------
        # 4. Trust score + explanation
        # -----------------------------
        overall_trust = self._compute_trust_score(
            bias_prob=bias_prob,
            sentiment_label=sentiment_label,
            sentiment_score=sentiment_score,
            num_propaganda_tags=len(propaganda_tags),
        )

        explanation_parts = [
            f"Bias model thinks this text is {bias_label} (biased probability={bias_prob:.2f}).",
            f"Overall sentiment: {sentiment_label} (confidence={sentiment_score:.2f}).",
        ]
        if propaganda_tags:
            explanation_parts.append(
                "Possible propaganda techniques: " + ", ".join(propaganda_tags) + "."
            )
        else:
            explanation_parts.append("No strong propaganda patterns detected.")
        explanation_parts.append(
            f"Combined into a trust score of {overall_trust:.2f} (0 = low, 1 = high)."
        )

        explanation = " ".join(explanation_parts)

        return {
            "bias_score": float(bias_prob),
            "bias_label": bias_label,
            "propaganda_tags": propaganda_tags,
            "emotional_tone": emotional_tone,
            "overall_trust_score": float(overall_trust),
            "explanation": explanation,
        }
