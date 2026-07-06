import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import cfg
from backend.app.inference import TwoStagePipeline
from backend.app.schemas import HealthResponse, PredictionItem, PredictionResponse

pipeline: TwoStagePipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    print("Loading two-stage pipeline...")
    pipeline = TwoStagePipeline()
    print("Pipeline ready.")
    yield
    pipeline = None


app = FastAPI(
    title="VisionGuard Crime Detection API",
    description="Two-stage pipeline: X3D-S binary gate → VideoMAE ViT-B 13-class classifier",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXT   = {".mp4", ".avi", ".mov", ".mkv"}
MAX_SIZE_MB   = 100


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    return HealthResponse(
        status="ok",
        model_loaded=pipeline is not None,
        num_classes=cfg.dataset.num_classes,
        classes=cfg.dataset.classes,
        pipeline="X3D-S → VideoMAE ViT-B",
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(file: UploadFile = File(...)):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded.")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {ext}")

    video_bytes = await file.read()
    if len(video_bytes) / (1024**2) > MAX_SIZE_MB:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_SIZE_MB} MB limit.")

    try:
        result = pipeline.predict(video_bytes)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    return PredictionResponse(
        is_normal         = result["is_normal"],
        stage1_confidence = result["stage1_confidence"],
        top_prediction    = result["top_prediction"],
        confidence        = result["confidence"],
        all_scores        = [
            PredictionItem(label=l, confidence=c)
            for l, c in (result["all_scores"] or [])
        ] if result["all_scores"] else None,
        inference_ms      = result["inference_ms"],
        stage1_ms         = result["stage1_ms"],
        stage2_ms         = result["stage2_ms"],
    )