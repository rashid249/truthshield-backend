# backend/models/image_analyzer.py

from fastapi import APIRouter, UploadFile, File
from backend.utils.inference import hf_image_inference

router = APIRouter(prefix="/analyze")


@router.post("/image")
async def analyze_image(file: UploadFile = File(...)):
    image_bytes = await file.read()

    nsfw = hf_image_inference(
        "falconsai/nsfw_image_detection",
        image_bytes
    )

    objects = hf_image_inference(
        "facebook/detr-resnet-50",
        image_bytes
    )

    return {
        "nsfw": nsfw,
        "objects": objects
    }
