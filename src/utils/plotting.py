from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt


def _read_metrics_csv(path: Path) -> Tuple[List[float], List[float], List[Optional[float]]]:
    epochs: List[float] = []
    train_losses: List[float] = []
    val_losses: List[Optional[float]] = []

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(float(row["epoch"]))
            train_losses.append(float(row["train_loss"]))
            val = row.get("val_loss", "")
            val_losses.append(float(val) if val not in ("", "nan", "NaN", None) else None)

    return epochs, train_losses, val_losses


def plot_losses(
    metrics_csv: Path,
    out_path: Path,
    title: str = "Loss over time",
    show: bool = False,
) -> None:
    """Plot training/validation losses from the metrics CSV."""
    if not metrics_csv.exists():
        return

    epochs, train_losses, val_losses = _read_metrics_csv(metrics_csv)

    plt.figure()
    plt.plot(epochs, train_losses, label="train")
    if any(v is not None for v in val_losses):
        xs = [e for e, v in zip(epochs, val_losses) if v is not None]
        ys = [v for v in val_losses if v is not None]
        plt.plot(xs, ys, label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    plt.close()
