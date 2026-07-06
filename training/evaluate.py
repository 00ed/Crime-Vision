import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import cfg
from dataset import get_dataloaders
from model import CrimeVideoClassifier, load_checkpoint


@torch.no_grad()
def run_evaluation(checkpoint_path: str = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _, _, test_loader, _ = get_dataloaders()

    model = CrimeVideoClassifier().to(device)
    ckpt_path = checkpoint_path or str(cfg.paths.checkpoint_dir / "best_model.pth")
    load_checkpoint(model, ckpt_path, device=device)
    model.eval()

    all_preds, all_labels = [], []

    for videos, labels in tqdm(test_loader, desc="Evaluating"):
        videos = videos.to(device, non_blocking=True)
        preds  = model(videos).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    print("\n" + "─"*60)
    print(classification_report(
        all_labels, all_preds,
        target_names=cfg.dataset.classes,
        digits=4,
    ))

    # Save JSON report
    report_dict = classification_report(
        all_labels, all_preds,
        target_names=cfg.dataset.classes,
        output_dict=True,
    )
    out = cfg.paths.checkpoint_dir / "evaluation_results.json"
    with open(out, "w") as f:
        json.dump(report_dict, f, indent=2)
    print(f"Results → {out}")

    # Confusion matrix
    cm  = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=cfg.dataset.classes,
                yticklabels=cfg.dataset.classes,
                cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix — DCSASS Crime Classification", fontsize=14)
    plt.tight_layout()

    cm_path = cfg.paths.checkpoint_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    print(f"Confusion matrix → {cm_path}")
    plt.show()

    acc = (all_preds == all_labels).mean()
    print(f"\nOverall test accuracy: {acc:.4f}")
    return acc


if __name__ == "__main__":
    run_evaluation()
