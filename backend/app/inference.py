"""
backend/app/inference.py
────────────────────────
Two-stage inference pipeline:
  Stage 1 → X3D-S binary:    Normal / Abnormal  (~10ms)
  Stage 2 → VideoMAE ViT-B:  13-class crime type (~200ms, only if Stage 1 = Abnormal)
"""

import sys
import time
import tempfile
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.config import cfg


class TwoStagePipeline:

    def __init__(
        self,
        binary_ckpt:  str | None = None,
        crime_ckpt:   str | None = None,
    ):
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        self.classes   = cfg.dataset.classes
        self.threshold = cfg.binary.abnormal_threshold

        # ── Stage 1: Binary classifier ────────────────────────────────────
        from training.binary_model import BinaryClassifier, load_binary_checkpoint
        self.binary_model = BinaryClassifier().to(self.device)
        b_ckpt = binary_ckpt or str(cfg.paths.checkpoint_dir / "binary_best.pth")
        load_binary_checkpoint(self.binary_model, b_ckpt, device=self.device)
        self.binary_model.eval()

        # ── Stage 2: Crime classifier ─────────────────────────────────────
        from training.model import CrimeVideoClassifier, load_checkpoint
        self.crime_model = CrimeVideoClassifier().to(self.device)
        c_ckpt = crime_ckpt or str(cfg.paths.checkpoint_dir / "best_model.pth")
        load_checkpoint(self.crime_model, c_ckpt, device=self.device)
        self.crime_model.eval()

        print(f"Two-stage pipeline ready on {self.device}")
        print(f"  Stage 1: X3D-S binary  (threshold={self.threshold})")
        print(f"  Stage 2: VideoMAE ViT-B 13-class")

    # ── Frame extraction ──────────────────────────────────────────────────

    def _extract(
        self,
        video_path: str,
        num_frames: int,
        frame_size: int,
    ) -> np.ndarray:
        from decord import VideoReader, cpu as decord_cpu

        vr = VideoReader(video_path, ctx=decord_cpu(0))
        total = len(vr)
        if total <= 0:
            raise ValueError("Video has no readable frames.")

        indices = np.linspace(0, total - 1, num_frames, dtype=int).tolist()
        frames  = vr.get_batch(indices).asnumpy()   # (T, H, W, 3) RGB

        frames = np.stack([
            cv2.resize(f, (frame_size, frame_size), interpolation=cv2.INTER_LINEAR)
            for f in frames
        ]).astype(np.float32)

        # Normalize
        frames /= 255.0
        mean = np.array(cfg.augmentation.mean, dtype=np.float32)
        std  = np.array(cfg.augmentation.std,  dtype=np.float32)
        return (frames - mean) / std   # (T, H, W, 3)

    def _to_tensor(self, frames: np.ndarray) -> torch.Tensor:
        """(T, H, W, C) → (1, C, T, H, W)"""
        return (
            torch.from_numpy(frames)
            .permute(3, 0, 1, 2)
            .unsqueeze(0)
            .to(self.device)
        )

    # ── Inference ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, video_bytes: bytes) -> dict:
        """
        Run two-stage inference on raw video bytes.

        Returns dict with:
            is_normal        bool
            stage1_confidence float   — confidence that clip is abnormal
            top_prediction   str | None
            confidence       float | None
            all_scores       list | None
            inference_ms     float
            stage1_ms        float
            stage2_ms        float | None
        """
        # Write to temp file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        try:
            return self._run(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def _run(self, video_path: str) -> dict:
        t_total = time.perf_counter()

        # ── Stage 1 ───────────────────────────────────────────────────────
        t1 = time.perf_counter()

        bcfg   = cfg.binary
        frames = self._extract(video_path, bcfg.num_frames, bcfg.frame_size)
        tensor = self._to_tensor(frames)

        logits1 = self.binary_model(tensor)           # (1, 2)
        probs1  = F.softmax(logits1, dim=1).squeeze() # (2,)
        abnormal_prob = probs1[1].item()
        is_normal     = abnormal_prob < self.threshold

        stage1_ms = (time.perf_counter() - t1) * 1000

        if is_normal:
            return {
                "is_normal":         True,
                "stage1_confidence": round(1 - abnormal_prob, 4),
                "top_prediction":    None,
                "confidence":        None,
                "all_scores":        None,
                "inference_ms":      round((time.perf_counter() - t_total) * 1000, 2),
                "stage1_ms":         round(stage1_ms, 2),
                "stage2_ms":         None,
            }

        # ── Stage 2 (only for abnormal clips) ────────────────────────────
        t2 = time.perf_counter()

        dcfg   = cfg.dataset
        frames2 = self._extract(video_path, dcfg.num_frames, dcfg.frame_size)
        tensor2 = self._to_tensor(frames2)

        logits2 = self.crime_model(tensor2)            # (1, 13)
        probs2  = F.softmax(logits2, dim=1).squeeze()  # (13,)

        top_idx  = probs2.argmax().item()
        top_conf = probs2[top_idx].item()
        top_label = self.classes[top_idx]

        all_scores = sorted(
            [(self.classes[i], round(probs2[i].item(), 4)) for i in range(len(self.classes))],
            key=lambda x: x[1],
            reverse=True,
        )

        stage2_ms = (time.perf_counter() - t2) * 1000

        return {
            "is_normal":         False,
            "stage1_confidence": round(abnormal_prob, 4),
            "top_prediction":    top_label,
            "confidence":        round(top_conf, 4),
            "all_scores":        all_scores,
            "inference_ms":      round((time.perf_counter() - t_total) * 1000, 2),
            "stage1_ms":         round(stage1_ms, 2),
            "stage2_ms":         round(stage2_ms, 2),
        }