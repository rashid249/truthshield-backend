# backend/models/image_analyzer.py

import io
from dataclasses import dataclass
from typing import Optional

import requests
from PIL import Image, ImageStat


@dataclass
class ImageAnalysisResult:
    ai_score: float
    label: str
    details: Optional[str] = None


class ImageAnalyzer:
    """
    Slightly smarter heuristic AI-image detector:

    - For URLs: downloads image
    - For uploaded files: reads raw bytes
    - Checks EXIF metadata (only for URL-based / some uploads)
    - Looks at simple statistics (smoothness / variance)

    Later you can replace the scoring logic with a real CNN/ViT model.
    """

    # -------- internal helper --------
    def _analyze_pil_image(self, img: Image.Image, has_exif: bool) -> ImageAnalysisResult:
        img = img.convert("RGB")

        # --- 1) Simple texture / variety heuristic ---
        small = img.resize((128, 128))
        stat = ImageStat.Stat(small)
        stdevs = stat.stddev
        avg_stdev = sum(stdevs) / len(stdevs)

        # Very smooth images are a bit more suspicious
        smoothness_suspicion = max(0.0, min(1.0, 80.0 / (avg_stdev + 1e-5)))

        # --- 2) Combine heuristics ---
        ai_score = 0.5
        details = []

        if has_exif:
            ai_score -= 0.2
            details.append("has EXIF (more like camera photo)")
        else:
            ai_score += 0.15
            details.append("no EXIF (could be AI or edited)")

        ai_score += 0.15 * smoothness_suspicion
        details.append(f"smoothness factor={smoothness_suspicion:.2f}")

        ai_score = float(max(0.0, min(1.0, ai_score)))

        if ai_score < 0.35:
            label = "likely_real"
        elif ai_score > 0.65:
            label = "possibly_ai_or_edited"
        else:
            label = "uncertain"

        return ImageAnalysisResult(ai_score=ai_score, label=label,
                                   details="; ".join(details))

    # -------- public: from URL --------
    def analyze_url(self, url: str) -> ImageAnalysisResult:
        try:
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content))

            exif = img.getexif()
            has_exif = exif is not None and len(exif) > 0

            return self._analyze_pil_image(img, has_exif=has_exif)

        except Exception as e:
            return ImageAnalysisResult(ai_score=0.5,
                                       label="error_loading_image",
                                       details=str(e))

    # -------- public: from raw bytes (file upload) --------
    def analyze_bytes(self, data: bytes) -> ImageAnalysisResult:
        try:
            img = Image.open(io.BytesIO(data))
            # EXIF often present on real photos, but might be stripped on upload
            exif = img.getexif()
            has_exif = exif is not None and len(exif) > 0
            return self._analyze_pil_image(img, has_exif=has_exif)
        except Exception as e:
            return ImageAnalysisResult(ai_score=0.5,
                                       label="error_loading_image",
                                       details=str(e))
