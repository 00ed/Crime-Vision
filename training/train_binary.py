"""
training/train_binary.py
────────────────────────
Train Stage 1: X3D-S binary (Normal / Abnormal) classifier.

Run from project root:
    python training/train_binary.py
"""

import sys
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import cfg
from training.binary_dataset import get_binary_dataloaders
from training.binary_model import (
    BinaryClassifier,
    save_binary_checkpoint,
)


# ── Train / eval steps ────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:02d} [train]", leave=False)
    for videos, labels in pbar:
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(videos)
        loss   = criterion(logits, labels)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), cfg.binary.grad_clip)
        optimizer.step()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += bs
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, split="val"):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for videos, labels in tqdm(loader, desc=f"  [{split}]", leave=False):
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(videos)
        loss   = criterion(logits, labels)

        bs = labels.size(0)
        total_loss += loss.item() * bs
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total   += bs
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    # Per-class accuracy
    normal_correct   = sum(p == l == 0 for p, l in zip(all_preds, all_labels))
    abnormal_correct = sum(p == l == 1 for p, l in zip(all_preds, all_labels))
    normal_total     = all_labels.count(0)
    abnormal_total   = all_labels.count(1)

    if split == "val":
        print(
            f"    Normal acc:   {normal_correct}/{normal_total} "
            f"({100*normal_correct/max(normal_total,1):.1f}%)"
        )
        print(
            f"    Abnormal acc: {abnormal_correct}/{abnormal_total} "
            f"({100*abnormal_correct/max(abnormal_total,1):.1f}%)"
        )

    return total_loss / total, correct / total


# ── Main ──────────────────────────────────────────────────────────────────────

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    bcfg = cfg.binary

    train_loader, val_loader, test_loader, class_weights = get_binary_dataloaders()
    class_weights = class_weights.to(device)

    model     = BinaryClassifier().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=bcfg.lr,
        weight_decay=bcfg.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=bcfg.epochs - bcfg.warmup_epochs,
        eta_min=1e-7,
    )

    writer    = SummaryWriter(log_dir=str(cfg.paths.log_dir / "binary"))
    best_ckpt = cfg.paths.checkpoint_dir / "binary_best.pth"

    best_val_acc      = 0.0
    epochs_no_improve = 0

    print(f"\nStage 1 training — {bcfg.epochs} epochs\n{'─'*60}")

    for epoch in range(1, bcfg.epochs + 1):

        # Unfreeze backbone after warmup
        if epoch == bcfg.warmup_epochs + 1:
            print(f"\n[Epoch {epoch}] Warmup done — unfreezing backbone")
            model.unfreeze_backbone()
            optimizer = AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=bcfg.lr * 0.1,
                weight_decay=bcfg.weight_decay,
            )
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=bcfg.epochs - epoch,
                eta_min=1e-7,
            )

        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, "val")
        scheduler.step()

        writer.add_scalars("Loss",     {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("Accuracy", {"train": train_acc,  "val": val_acc},  epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        print(
            f"Epoch {epoch:02d}/{bcfg.epochs} | "
            f"train {train_loss:.4f}/{train_acc:.4f} | "
            f"val {val_loss:.4f}/{val_acc:.4f} | "
            f"{time.time()-t0:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc      = val_acc
            epochs_no_improve = 0
            save_binary_checkpoint(model, optimizer, epoch, val_acc, best_ckpt)
        else:
            epochs_no_improve += 1
            print(f"  No improvement {epochs_no_improve}/{bcfg.patience}")

        if epochs_no_improve >= bcfg.patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    writer.close()


def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    _, _, test_loader, class_weights = get_binary_dataloaders()
    class_weights = class_weights.to(device)

    model     = BinaryClassifier().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_ckpt = cfg.paths.checkpoint_dir / "binary_best.pth"
    if not best_ckpt.exists():
        raise RuntimeError(f"Checkpoint not found: {best_ckpt}")
    print(f"\nLoading checkpoint: {best_ckpt}")
    checkpoint = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_loss, test_acc = evaluate(model, test_loader, criterion, device, "test")
    print(f"\nTest results — loss: {test_loss:.4f}, acc: {test_acc:.4f}")


if __name__ == "__main__":
    #train()
    test()
