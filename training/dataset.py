import os
import sys
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import List, Tuple, Optional
from decord import VideoReader, cpu as decord_cpu

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import cfg

# ── Transforms ────────────────────────────────────────────────────────────────

def normalize_frames(frames: np.ndarray) -> np.ndarray:
    frames = frames.astype(np.float32) / 255.0
    mean = np.array(cfg.augmentation.mean, dtype=np.float32)
    std  = np.array(cfg.augmentation.std,  dtype=np.float32)
    return (frames - mean) / std


def augment_frames(frames: np.ndarray) -> np.ndarray:
    """Spatial augmentations applied identically to every frame in a clip."""
    T, H, W, C = frames.shape
    aug = cfg.augmentation

    # Random horizontal flip
    if random.random() < aug.flip_prob:
        frames = frames[:, :, ::-1, :].copy()

    # Random crop (scale jitter)
    scale = random.uniform(aug.scale_min, aug.scale_max)
    crop_h, crop_w = int(H * scale), int(W * scale)
    top  = random.randint(0, H - crop_h)
    left = random.randint(0, W - crop_w)
    frames = frames[:, top:top+crop_h, left:left+crop_w, :]
    frames = np.stack([
        cv2.resize(f, (W, H), interpolation=cv2.INTER_LINEAR) for f in frames
    ])

    # Brightness / contrast jitter
    alpha = random.uniform(aug.brightness_min, aug.brightness_max)
    beta  = random.uniform(-aug.contrast_delta, aug.contrast_delta)
    frames = np.clip(frames * alpha + beta, 0, 255).astype(np.uint8)

    return frames


def extract_frames(
    video_path: str,
    num_frames: int,
    training: bool,
) -> Optional[np.ndarray]:
    try:
        vr = VideoReader(video_path, ctx=decord_cpu(0))
    except Exception:
        return None

    total = len(vr)
    if total <= 0:
        return None

    indices = np.linspace(0, total - 1, num_frames, dtype=int).tolist()
    frames = vr.get_batch(indices).asnumpy()  # (T, H, W, 3) — already RGB

    size = cfg.dataset.frame_size
    frames = np.stack([
        cv2.resize(f, (size, size), interpolation=cv2.INTER_LINEAR)
        for f in frames
    ])

    if training:
        frames = augment_frames(frames)

    return normalize_frames(frames)


# ── Dataset ───────────────────────────────────────────────────────────────────

def collect_samples(data_dir: Path) -> List[Tuple[str, int]]:
    """Walk DCSASS folder, skip non-class dirs (e.g. Labels), return (path, idx) pairs."""
    skip = set(cfg.dataset.skip_folders)
    samples = []

    for class_name, class_idx in cfg.dataset.class_to_idx.items():
        if class_name in skip:
            continue
        class_dir = data_dir / class_name
        if not class_dir.exists():
            print(f"[WARNING] Class folder not found: {class_dir}")
            continue
        # Iterate over everything directly inside the class folder (e.g., 'Abuse')
        for item in class_dir.iterdir():
            
            # If the item is a folder (even if it is named like '.mp4') step into it
            if item.is_dir():
                for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
                    # Use .glob() to find the actual clips inside this sub-folder
                    for video_path in item.glob(ext):
                        if video_path.is_file():
                            samples.append((str(video_path), class_idx))

    if not samples:
        raise RuntimeError(
            f"No videos found in {data_dir}.\n"
            "Check that DCSASS_PATH in your .env points to the correct folder."
        )

    print(f"Found {len(samples)} total videos across {cfg.dataset.num_classes} classes")
    return samples


def split_samples(
    samples: List[Tuple[str, int]],
) -> Tuple[List, List, List]:
    """Stratified train / val / test split."""
    paths, labels = zip(*samples)
    ds = cfg.dataset

    paths_tv, paths_test, labels_tv, labels_test = train_test_split(
        paths, labels,
        test_size=ds.test_split,
        stratify=labels,
        random_state=ds.seed,
    )
    val_ratio = ds.val_split / (1 - ds.test_split)
    paths_train, paths_val, labels_train, labels_val = train_test_split(
        paths_tv, labels_tv,
        test_size=val_ratio,
        stratify=labels_tv,
        random_state=ds.seed,
    )

    return (
        list(zip(paths_train, labels_train)),
        list(zip(paths_val,   labels_val)),
        list(zip(paths_test,  labels_test)),
    )


class DCSSASSDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], training: bool = True):
        self.samples   = samples
        self.num_frames = cfg.dataset.num_frames
        self.training  = training

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        video_path, label = self.samples[idx]
        frames = extract_frames(video_path, self.num_frames, self.training)

        if frames is None:
            frames = np.zeros(
                (self.num_frames, cfg.dataset.frame_size, cfg.dataset.frame_size, 3),
                dtype=np.float32,
            )

        # (T, H, W, C) → (C, T, H, W)
        tensor = torch.from_numpy(frames).permute(3, 0, 1, 2)
        return tensor, label


# ── Class weights ─────────────────────────────────────────────────────────────

def compute_class_weights(samples: List[Tuple[str, int]]) -> torch.Tensor:
    counts = np.zeros(cfg.dataset.num_classes, dtype=np.float32)
    for _, label in samples:
        counts[label] += 1
    counts = np.where(counts == 0, 1, counts)
    weights = 1.0 / counts
    weights = weights / weights.sum() * cfg.dataset.num_classes
    return torch.tensor(weights, dtype=torch.float32)


# ── DataLoader factory ────────────────────────────────────────────────────────

def get_dataloaders() -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    data_dir = cfg.paths.data_dir
    samples  = collect_samples(data_dir)
    train_s, val_s, test_s = split_samples(samples)

    print(f"Split → train: {len(train_s)} | val: {len(val_s)} | test: {len(test_s)}")

    train_ds = DCSSASSDataset(train_s, training=True)
    val_ds   = DCSSASSDataset(val_s,   training=False)
    test_ds  = DCSSASSDataset(test_s,  training=False)

    bs  = cfg.training.batch_size
    nw  = cfg.training.num_workers

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=nw, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                              num_workers=nw, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False,
                              num_workers=nw, pin_memory=True)

    return train_loader, val_loader, test_loader, compute_class_weights(train_s)
