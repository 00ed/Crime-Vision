from pydantic import BaseModel
from typing import List, Optional


class PredictionItem(BaseModel):
    label:      str
    confidence: float


class PredictionResponse(BaseModel):
    # Stage 1 result
    is_normal:         bool
    stage1_confidence: float      # P(abnormal) if abnormal, P(normal) if normal

    # Stage 2 result — None if clip was normal
    top_prediction:    Optional[str]
    confidence:        Optional[float]
    all_scores:        Optional[List[PredictionItem]]

    # Timing
    inference_ms:      float
    stage1_ms:         float
    stage2_ms:         Optional[float]


class HealthResponse(BaseModel):
    status:       str
    model_loaded: bool
    num_classes:  int
    classes:      List[str]
    pipeline:     str