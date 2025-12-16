# backend/app.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.models.text_analyzer import router as text_router
from backend.models.image_analyzer import router as image_router

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount all routes
app.include_router(text_router)
app.include_router(image_router)


@app.get("/")
def root():
    return {"status": "TruthShield + HF Router API running"}
