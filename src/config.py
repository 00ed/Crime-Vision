"""
src/config.py
─────────────
Single entry point for all configuration across the project.

Usage:
    from src.config import cfg

    cfg.dataset.classes
    cfg.binary.abnormal_threshold
    cfg.training.lr
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, field_validator, model_validator

load_dotenv()

_CONFIGS_DIR  = Path(__file__).resolve().parent.parent / "configs"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── Pydantic models ───────────────────────────────────────────────────────────

class PathsConfig(BaseModel):
    data_dir:       Path
    checkpoint_dir: Path
    log_dir:        Path

    @model_validator(mode="after")
    def create_dirs(self) -> "PathsConfig":
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        return self


class DatasetConfig(BaseModel):
    classes:      List[str]
    skip_folders: List[str]
    num_frames:   int
    frame_size:   int
    val_split:    float
    test_split:   float
    seed:         int

    @field_validator("val_split", "test_split")
    @classmethod
    def valid_split(cls, v: float) -> float:
        assert 0 < v < 1, "Splits must be between 0 and 1"
        return v

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def class_to_idx(self) -> dict[str, int]:
        return {c: i for i, c in enumerate(self.classes)}

    @property
    def idx_to_class(self) -> dict[int, str]:
        return {i: c for c, i in self.class_to_idx.items()}


class BinaryConfig(BaseModel):
    """Stage 1 — X3D-S binary (Normal / Abnormal) classifier."""
    labels_dir:          Path
    col_clip:            str
    col_label:           str
    model_name:          str
    use_x3d:             bool
    num_frames:          int
    frame_size:          int
    dropout:             float
    batch_size:          int
    epochs:              int
    lr:                  float
    weight_decay:        float
    warmup_epochs:       int
    patience:            int
    grad_clip:           float
    num_workers:         int
    val_split:           float
    test_split:          float
    abnormal_threshold:  float


class ModelConfig(BaseModel):
    name:           str
    dropout:        float
    freeze_encoder: bool


class TrainingConfig(BaseModel):
    batch_size:          int
    epochs:              int
    lr:                  float
    weight_decay:        float
    warmup_epochs:       int
    patience:            int
    unfreeze_layers:     int
    unfreeze_lr_factor:  float
    grad_clip:           float
    num_workers:         int


class AugmentationConfig(BaseModel):
    mean:            List[float]
    std:             List[float]
    flip_prob:       float
    scale_min:       float
    scale_max:       float
    brightness_min:  float
    brightness_max:  float
    contrast_delta:  float


class AppConfig(BaseModel):
    paths:        PathsConfig
    dataset:      DatasetConfig
    binary:       BinaryConfig
    model:        ModelConfig
    training:     TrainingConfig
    augmentation: AugmentationConfig


# ── YAML loader with env-var interpolation ────────────────────────────────────

def _interpolate_env(value: str) -> str:
    pattern = re.compile(r"\$\{([^}]+)\}")
    def replacer(match: re.Match) -> str:
        var = match.group(1)
        val = os.environ.get(var)
        if val is None:
            raise EnvironmentError(
                f"Environment variable '{var}' is not set.\n"
                f"Copy .env.example → .env and fill in {var}."
            )
        return val
    return pattern.sub(replacer, value)


def _resolve(raw: dict, root: Path) -> dict:
    resolved = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            resolved[key] = _resolve(value, root)
        elif isinstance(value, str):
            interpolated = _interpolate_env(value)
            p = Path(interpolated)
            if not p.is_absolute():
                p = root / p
            resolved[key] = str(p)
        else:
            resolved[key] = value
    return resolved


def load_config(config_path: Path | None = None) -> AppConfig:
    path = config_path or (_CONFIGS_DIR / "config.yaml")
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    raw["paths"]  = _resolve(raw["paths"],  _PROJECT_ROOT)
    raw["binary"] = _resolve(raw["binary"], _PROJECT_ROOT)
    return AppConfig(**raw)


# ── Singleton ─────────────────────────────────────────────────────────────────
cfg: AppConfig = load_config()