from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from utils.schema import (
    TextRequest,
    TextAnalysisResponse,
    ImageRequest,
    ImageAnalysisResponse,
)
from models.text_analyzer import TextBiasAnalyzer
from models.image_analyzer import ImageAnalyzer

app = FastAPI(title="TruthShield Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # for dev; tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

text_analyzer = TextBiasAnalyzer()
image_analyzer = ImageAnalyzer()


@app.get("/")
async def root():
    return {"status": "TruthShield backend running"}


@app.post("/analyze_text", response_model=TextAnalysisResponse)
async def analyze_text(req: TextRequest):
    result = text_analyzer.analyze(req.text)
    return TextAnalysisResponse(**result)


@app.post("/analyze_image", response_model=ImageAnalysisResponse)
async def analyze_image(req: ImageRequest):
    result = image_analyzer.analyze_url(req.url)
    return ImageAnalysisResponse(
        ai_score=result.ai_score,
        label=result.label,
        details=result.details,
    )


@app.post("/analyze_image_file", response_model=ImageAnalysisResponse)
async def analyze_image_file(file: UploadFile = File(...)):
    data = await file.read()
    result = image_analyzer.analyze_bytes(data)
    return ImageAnalysisResponse(
        ai_score=result.ai_score,
        label=result.label,
        details=result.details,
    )
