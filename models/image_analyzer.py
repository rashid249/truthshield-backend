# backend/models/image_analyzer.py

import io
import os
from dataclasses import dataclass
from typing import Optional

import requests
from PIL import Image, ImageStat


HF_API_KEY = os.getenv("HF_API_KEY")
HF_API_URL = "https://api-inference.huggingface.co/models/omni-moderation-latest"  
# This model returns: nsfw_score, fake_score, modification_score


headers = {"Authorization": f"Bearer {HF_API_KEY}"}


@dataclass
class ImageAnalysisResult:
    ai_score: float
    label: str
    nsfw_score: Optional[float] = None
    manipulation_score: Optional[float] = None
    details: Optional[str] = None


class ImageAnalyzer:

    # --------------------------
    # 1) HuggingFace Remote AI
    # --------------------------
    def _huggingface_analyze(self, image_bytes: bytes) -> Optional[dict]:
        try:
            response = requests.post(
                HF_API_URL,
                headers=headers,
                data=image_bytes,
                timeout=12
            )

            if response.status_code != 200:
                return None

            return response.json()

        except Exception:
            return None

    # -------------------------
    # 2) Local heuristic fallback
    # -------------------------
    def _heuristic_analyze(self, img: Image.Image, has_exif: bool) -> ImageAnalysisResult:
        img = img.convert("RGB")
        small = img.resize((128, 128))
        stat = ImageStat.Stat(small)

        stdevs = stat.stddev
        avg_stdev = sum(stdevs) / len(stdevs)
        smoothness_suspicion = max(0.0, min(1.0, 80.0 / (avg_stdev + 1e-5)))

        ai_score = 0.5
        details = []

        if has_exif:
            ai_score -= 0.2
            details.append("Has EXIF (more like a real camera image)")
        else:
            ai_score += 0.15
            details.append("No EXIF (may be AI-generated or edited)")

        ai_score += 0.15 * smoothness_suspicion
        details.append(f"smoothness={smoothness_suspicion:.2f}")

        ai_score = float(max(0.0, min(1.0, ai_score)))

        if ai_score < 0.35:
            label = "likely_real"
        elif ai_score > 0.65:
            label = "possibly_ai_or_edited"
        else:
            label = "uncertain"

        return ImageAnalysisResult(
            ai_score=ai_score,
            label=label,
            nsfw_score=None,
            manipulation_score=None,
            details="; ".join(details)
        )

    # -------------------------
    # 3) Public: analyze bytes
    # -------------------------
    def analyze_bytes(self, data: bytes) -> ImageAnalysisResult:
        # 1) Try HuggingFace AI first
        hf_result = self._huggingface_analyze(data)

        if hf_result:
            ai_score = float(hf_result.get("fake_score", 0.0))
            nsfw_score = float(hf_result.get("nsfw_score", 0.0))
            manipulation_score = float(hf_result.get("modification_score", 0.0))

            if ai_score > 0.6:
                label = "ai_generated_or_deepfake"
            elif manipulation_score > 0.45:
                label = "digitally_modified"
            elif nsfw_score > 0.8:
                label = "unsafe_content"
            else:
                label = "likely_real"

            return ImageAnalysisResult(
                ai_score=ai_score,
                label=label,
                nsfw_score=nsfw_score,
                manipulation_score=manipulation_score,
                details="HuggingFace model analysis"
            )

        # 2) If HF API fails → fallback to heuristic
        try:
            img = Image.open(io.BytesIO(data))
            exif = img.getexif()
            has_exif = exif is not None and len(exif) > 0
            return self._heuristic_analyze(img, has_exif)

        except Exception as e:
            return ImageAnalysisResult(
                ai_score=0.5,
                label="error_loading_image",
                details=str(e)
            )

    # -------------------------
    # 4) Public: analyze URL
    # -------------------------
    def analyze_url(self, url: str) -> ImageAnalysisResult:
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            return self.analyze_bytes(resp.content)
        except Exception as e:
            return ImageAnalysisResult(
                ai_score=0.5,
                label="error_loading_image",
                details=str(e)
            )
