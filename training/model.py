import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
from transformers import VideoMAEModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import cfg


class CrimeVideoClassifier(nn.Module):
    """
    VideoMAE ViT-B fine-tuned for 13-class DCSASS crime classification.
    Encoder is frozen during warmup then gradually unfrozen.
    """

    def __init__(self):
        super().__init__()
        mcfg = cfg.model

        print(f"Loading VideoMAE backbone: {mcfg.name}")
        self.encoder = VideoMAEModel.from_pretrained(
            mcfg.name,
            token=os.getenv("HF_TOKEN") or None,
        )
        hidden_size  = self.encoder.config.hidden_size  # 768 for ViT-B

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(mcfg.dropout),
            nn.Linear(hidden_size, cfg.dataset.num_classes),
        )

        if mcfg.freeze_encoder:
            self._freeze_encoder()

    def _freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.encoder.layernorm.parameters():
            param.requires_grad = True
        print("Encoder frozen — only classifier head is trainable.")

    def unfreeze_encoder(self, num_layers: int | None = None):
        n = num_layers or cfg.training.unfreeze_layers
        all_layers = self.encoder.encoder.layer
        total = len(all_layers)
        for i, layer in enumerate(all_layers):
            if i >= total - n:
                for param in layer.parameters():
                    param.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Unfroze last {n} encoder blocks. Trainable params: {trainable:,}")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # (B, C, T, H, W) → (B, T, C, H, W) for VideoMAE
        x = pixel_values.permute(0, 2, 1, 3, 4)
        outputs = self.encoder(pixel_values=x)
        pooled  = outputs.last_hidden_state.mean(dim=1)
        return self.classifier(pooled)


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, scheduler, epoch: int, val_acc: float, path: Path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "val_acc": val_acc,
        "classes": cfg.dataset.classes,
        "config": {
            "model_name": cfg.model.name,
            "num_classes": cfg.dataset.num_classes,
        },
    }, path)
    print(f"Checkpoint saved → {path}  (val_acc={val_acc:.4f})")


def load_checkpoint(model, path: Path, optimizer=None, scheduler=None, device="cuda"):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']}  (val_acc={ckpt['val_acc']:.4f})")
    return ckpt
