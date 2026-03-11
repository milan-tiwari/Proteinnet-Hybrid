from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def unwrap_ddp(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def save_checkpoint(
    *,
    ckpt_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    scaler: Optional[torch.cuda.amp.GradScaler],
    epoch: int,
    global_step: int,
    config: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "epoch": epoch,
        "global_step": global_step,
        "model": unwrap_ddp(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
    }
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    if extra:
        payload["extra"] = extra

    torch.save(payload, ckpt_path)


def load_checkpoint(
    ckpt_path: Path,
    *,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    payload = torch.load(ckpt_path, map_location=map_location)

    unwrap_ddp(model).load_state_dict(payload["model"], strict=True)

    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and "scaler" in payload:
        scaler.load_state_dict(payload["scaler"])

    return payload


def find_latest_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    """Return ckpt_dir/latest.pt if it exists, else None."""
    p = ckpt_dir / "latest.pt"
    return p if p.exists() else None


def rotate_checkpoints(ckpt_dir: Path, keep_last_k: int) -> None:
    """Keep only the latest K epoch checkpoints (best.pt and latest.pt are kept)."""
    if keep_last_k <= 0:
        return

    candidates = sorted(ckpt_dir.glob("epoch_*.pt"))
    if len(candidates) <= keep_last_k:
        return

    to_delete = candidates[:-keep_last_k]
    for p in to_delete:
        try:
            p.unlink()
        except OSError:
            pass
