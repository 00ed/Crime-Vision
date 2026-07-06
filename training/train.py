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
from dataset import get_dataloaders
from model import CrimeVideoClassifier, save_checkpoint


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

        nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
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

    for videos, labels in tqdm(loader, desc=f"  [{split}]", leave=False):
        videos = videos.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(videos)
        loss   = criterion(logits, labels)

        bs = labels.size(0)
        total_loss += loss.item() * bs
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += bs

    return total_loss / total, correct / total


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    tcfg = cfg.training

    train_loader, val_loader, test_loader, class_weights = get_dataloaders()
    class_weights = class_weights.to(device)

    model     = CrimeVideoClassifier().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=tcfg.lr,
        weight_decay=tcfg.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=tcfg.epochs - tcfg.warmup_epochs,
        eta_min=1e-7,
    )

    writer = SummaryWriter(log_dir=str(cfg.paths.log_dir))

    best_val_acc      = 0.0
    epochs_no_improve = 0
    best_ckpt         = cfg.paths.checkpoint_dir / "best_model.pth"

    print(f"\nStarting training — {tcfg.epochs} epochs\n{'─'*60}")

    for epoch in range(1, tcfg.epochs + 1):

        # ── Gradual unfreeze after warmup ──────────────────────────────────
        if epoch == tcfg.warmup_epochs + 1:
            print(f"\n[Epoch {epoch}] Warmup done — unfreezing last {tcfg.unfreeze_layers} blocks")
            model.unfreeze_encoder()
            optimizer = AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=tcfg.lr * tcfg.unfreeze_lr_factor,
                weight_decay=tcfg.weight_decay,
            )
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=tcfg.epochs - epoch,
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
            f"Epoch {epoch:02d}/{tcfg.epochs} | "
            f"train {train_loss:.4f}/{train_acc:.4f} | "
            f"val {val_loss:.4f}/{val_acc:.4f} | "
            f"{time.time()-t0:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc      = val_acc
            epochs_no_improve = 0
            save_checkpoint(model, optimizer, scheduler, epoch, val_acc, best_ckpt)
        else:
            epochs_no_improve += 1
            print(f"  No improvement {epochs_no_improve}/{tcfg.patience}")

        if epochs_no_improve >= tcfg.patience:
            print(f"\nEarly stopping at epoch {epoch} — best val_acc={best_val_acc:.4f}")
            break

    writer.close()
    print(f"\nDone. Best val_acc = {best_val_acc:.4f} → {best_ckpt}")
    return best_ckpt


if __name__ == "__main__":
    train()
