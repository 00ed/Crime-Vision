"""
training/binary_dataset.py
──────────────────────────
Reads the 13 per-class CSVs from the Labels folder.
Each CSV has rows like:
    Abuse001_x264_1, Abuse, 1
    Abuse001_x264_2, Abuse, 0

The third column (0/1) is the binary label:
    0 = normal segment
    1 = anomaly present

We resolve each clip name back to its video file on disk,
then build a balanced Dataset for Stage 1 binary training.
"""

import sys
import csv
import random
import numpy as np
import torch
import cv2
from pathlib import Path
from typing import List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from decord import VideoReader, cpu as decord_cpu

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import cfg


# ── CSV parsing ───────────────────────────────────────────────────────────────

def load_labels_from_csvs() -> List[Tuple[str, int]]:
    """
    Walk every CSV in the Labels folder.
    Returns list of (video_path, binary_label) tuples.
    binary_label: 0 = normal, 1 = abnormal
    """
    bcfg      = cfg.binary
    labels_dir = bcfg.labels_dir
    data_dir   = cfg.paths.data_dir

    all_samples: List[Tuple[str, int]] = []
    skipped = 0

    for csv_path in sorted(labels_dir.glob("*.csv")):
        class_name = csv_path.stem          # e.g. "Abuse" from "Abuse.csv"
        class_dir  = data_dir / class_name  # e.g. .../DCSASS Dataset/Abuse/

        if not class_dir.exists():
            print(f"[WARNING] No folder found for class: {class_name}, skipping.")
            continue

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            # REMOVED: header = next(reader, None) <-- Your CSVs don't have headers!

            for row in reader:
                if len(row) < 2:
                    continue

                clip_name  = row[0].strip()   # e.g. Abuse001_x264_1
                raw_label  = row[-1].strip()

                try:
                    label = int(raw_label)    # 0 or 1
                except ValueError:
                    continue

                parts      = clip_name.rsplit("_", 1)
                if len(parts) != 2:
                    skipped += 1
                    continue

                base_video = parts[0]         # e.g. Abuse001_x264
                
                # ADDED .mp4: We must add .mp4 because your folders are named that way
                video_dir  = class_dir / f"{base_video}.mp4"

                if not video_dir.exists():
                    skipped += 1
                    continue

                # Find the clip file inside the subfolder
                clip_file = None
                for ext in (".mp4", ".avi", ".mov", ".mkv"):
                    candidate = video_dir / f"{clip_name}{ext}"
                    if candidate.exists():
                        clip_file = candidate
                        break

                # Also try without extension matching exactly
                if clip_file is None:
                    for f_path in video_dir.iterdir():
                        if f_path.stem == clip_name:
                            clip_file = f_path
                            break

                if clip_file is None:
                    skipped += 1
                    continue

                all_samples.append((str(clip_file), label))

    normal_count   = sum(1 for _, l in all_samples if l == 0)
    abnormal_count = sum(1 for _, l in all_samples if l == 1)

    print(f"Binary dataset loaded:")
    print(f"  Total clips : {len(all_samples)}")
    print(f"  Normal  (0) : {normal_count}")
    print(f"  Abnormal(1) : {abnormal_count}")
    print(f"  Skipped     : {skipped}")

    if not all_samples:
        raise RuntimeError(
            "No samples found. Check that DCSASS_PATH and the Labels folder "
            "structure match what binary_dataset.py expects."
        )

    return all_samples


def split_binary_samples(
    samples: List[Tuple[str, int]],
) -> Tuple[List, List, List]:
    """Stratified train / val / test split."""
    bcfg   = cfg.binary
    paths, labels = zip(*samples)

    paths_tv, paths_test, labels_tv, labels_test = train_test_split(
        paths, labels,
        test_size=bcfg.test_split,
        stratify=labels,
        random_state=cfg.dataset.seed,
    )
    val_ratio = bcfg.val_split / (1 - bcfg.test_split)
    paths_train, paths_val, labels_train, labels_val = train_test_split(
        paths_tv, labels_tv,
        test_size=val_ratio,
        stratify=labels_tv,
        random_state=cfg.dataset.seed,
    )
    return (
        list(zip(paths_train, labels_train)),
        list(zip(paths_val,   labels_val)),
        list(zip(paths_test,  labels_test)),
    )


# ── Frame extraction ──────────────────────────────────────────────────────────

def _augment(frames: np.ndarray) -> np.ndarray:
    """Light augmentation — only for training."""
    aug = cfg.augmentation
    if random.random() < aug.flip_prob:
        frames = frames[:, :, ::-1, :].copy()
    alpha = random.uniform(aug.brightness_min, aug.brightness_max)
    beta  = random.uniform(-aug.contrast_delta, aug.contrast_delta)
    frames = np.clip(frames * alpha + beta, 0, 255).astype(np.uint8)
    return frames


def _normalize(frames: np.ndarray) -> np.ndarray:
    frames = frames.astype(np.float32) / 255.0
    mean = np.array(cfg.augmentation.mean, dtype=np.float32)
    std  = np.array(cfg.augmentation.std,  dtype=np.float32)
    return (frames - mean) / std


def extract_frames_binary(
    video_path: str,
    training: bool,
) -> Optional[np.ndarray]:
    """Returns (T, H, W, 3) float32 normalized or None on failure."""
    bcfg = cfg.binary
    try:
        vr = VideoReader(video_path, ctx=decord_cpu(0))
    except Exception:
        return None

    total = len(vr)
    if total <= 0:
        return None

    indices = np.linspace(0, total - 1, bcfg.num_frames, dtype=int).tolist()
    frames  = vr.get_batch(indices).asnumpy()   # (T, H, W, 3) RGB

    size = bcfg.frame_size
    frames = np.stack([
        cv2.resize(f, (size, size), interpolation=cv2.INTER_LINEAR)
        for f in frames
    ])

    if training:
        frames = _augment(frames)

    return _normalize(frames)


# ── Dataset ───────────────────────────────────────────────────────────────────

class BinaryDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], training: bool = True):
        self.samples  = samples
        self.training = training
        self.bcfg     = cfg.binary

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        video_path, label = self.samples[idx]
        frames = extract_frames_binary(video_path, self.training)

        if frames is None:
            frames = np.zeros(
                (self.bcfg.num_frames, self.bcfg.frame_size, self.bcfg.frame_size, 3),
                dtype=np.float32,
            )

        # (T, H, W, C) → (C, T, H, W)
        tensor = torch.from_numpy(frames).permute(3, 0, 1, 2)
        return tensor, label


# ── Class weights ─────────────────────────────────────────────────────────────

def compute_binary_weights(samples: List[Tuple[str, int]]) -> torch.Tensor:
    """Handle class imbalance between normal and abnormal clips."""
    counts = np.zeros(2, dtype=np.float32)
    for _, label in samples:
        counts[label] += 1
    counts  = np.where(counts == 0, 1, counts)
    weights = 1.0 / counts
    weights = weights / weights.sum() * 2
    return torch.tensor(weights, dtype=torch.float32)


# ── DataLoader factory ────────────────────────────────────────────────────────

def get_binary_dataloaders() -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    bcfg    = cfg.binary
    samples = load_labels_from_csvs()
    train_s, val_s, test_s = split_binary_samples(samples)

    print(f"Split → train: {len(train_s)} | val: {len(val_s)} | test: {len(test_s)}")

    train_ds = BinaryDataset(train_s, training=True)
    val_ds   = BinaryDataset(val_s,   training=False)
    test_ds  = BinaryDataset(test_s,  training=False)

    train_loader = DataLoader(
        train_ds, batch_size=bcfg.batch_size, shuffle=True,
        num_workers=bcfg.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bcfg.batch_size, shuffle=False,
        num_workers=bcfg.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=bcfg.batch_size, shuffle=False,
        num_workers=bcfg.num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader, compute_binary_weights(train_s)
