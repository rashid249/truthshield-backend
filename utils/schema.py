# backend/utils/schema.py
from pydantic import BaseModel
from typing import List, Dict, Optional


class TextRequest(BaseModel):
  text: str


class TextAnalysisResponse(BaseModel):
  bias_score: float
  bias_label: str
  propaganda_tags: List[str]
  emotional_tone: Dict[str, float]
  overall_trust_score: float
  explanation: str


class ImageRequest(BaseModel):
  url: str


class ImageAnalysisResponse(BaseModel):
  ai_score: float
  label: str
  details: Optional[str] = None
