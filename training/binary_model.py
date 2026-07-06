"""
training/binary_model.py
────────────────────────
Stage 1 gatekeeper: X3D-S binary classifier (Normal / Abnormal).

X3D-S: 3.8M parameters, ~10ms inference — designed to be fast.
If torchvision X3D-S is unavailable, falls back to a lightweight
R3D-18 binary head (same interface, slightly heavier).
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import cfg


class BinaryClassifier(nn.Module):
    """
    X3D-S backbone with a 2-class (Normal / Abnormal) head.
    Backbone is frozen initially, unfrozen after warmup.
    """

    def __init__(self):
        super().__init__()
        bcfg = cfg.binary

        if bcfg.use_x3d:
            self.backbone, in_features = self._build_x3d()
            self.model_tag = "X3D-S"
        else:
            self.backbone, in_features = self._build_r3d()
            self.model_tag = "R3D-18"

        self.head = nn.Sequential(
            nn.Dropout(bcfg.dropout),
            nn.Linear(in_features, 2),
        )

        self._freeze_backbone()
        print(f"Binary classifier: {self.model_tag} + 2-class head")

    def _build_x3d(self):
        """Load X3D-S from torchvision."""
        try:
            from torchvision.models.video import x3d_s, X3D_S_Weights
            base = x3d_s(weights=X3D_S_Weights.DEFAULT)
            # X3D-S final projection: 2048 → use the penultimate feature
            in_features = base.blocks[-1].proj.in_features
            # Remove the final classification head — keep feature extractor
            base.blocks[-1].proj = nn.Identity()
            base.blocks[-1].activation = nn.Identity()
            return base, in_features
        except Exception as e:
            print(f"[WARNING] X3D-S unavailable ({e}), falling back to R3D-18")
            return self._build_r3d()

    def _build_r3d(self):
        """Fallback: R3D-18 as binary backbone."""
        from torchvision.models.video import r3d_18, R3D_18_Weights
        base = r3d_18(weights=R3D_18_Weights.DEFAULT)
        in_features = base.fc.in_features
        base.fc = nn.Identity()
        return base, in_features

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        print(f"  Backbone frozen — only head trainable.")

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Backbone unfrozen. Trainable params: {total:,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, H, W)
        Returns:
            logits: (B, 2)
        """
        features = self.backbone(x)     # (B, in_features)
        return self.head(features)      # (B, 2)


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_binary_checkpoint(
    model: BinaryClassifier,
    optimizer,
    epoch: int,
    val_acc: float,
    path: Path,
):
    torch.save({
        "epoch":              epoch,
        "model_state_dict":   model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_acc":            val_acc,
        "model_tag":          model.model_tag,
        "threshold":          cfg.binary.abnormal_threshold,
    }, path)
    print(f"Binary checkpoint saved → {path}  (val_acc={val_acc:.4f})")


def load_binary_checkpoint(
    model: BinaryClassifier,
    path: Path,
    device: str = "cuda",
) -> dict:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    print(
        f"Loaded binary checkpoint from epoch {ckpt['epoch']}  "
        f"(val_acc={ckpt['val_acc']:.4f}, model={ckpt.get('model_tag','?')})"
    )
    return ckpt
