#!/usr/bin/env python


from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
from tqdm import tqdm

# Make sure `import src...` works even if launched from another directory.
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.collate import collate_batch
from src.data.memmap_dataset import ProteinNetMemmapDataset
from src.models.transformer import ProteinTransformer, ProteinTransformerConfig
from src.utils.checkpointing import (
    find_latest_checkpoint,
    load_checkpoint,
    rotate_checkpoints,
    save_checkpoint,
    unwrap_ddp,
)
from src.utils.distributed import DistInfo, cleanup_distributed, init_distributed, is_main_process
from src.utils.metrics import batch_rmsd
from src.utils.plotting import plot_losses
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a Transformer on ProteinNet (hybrid-ready).")
    p.add_argument("--config", type=Path, required=True, help="Path to a YAML config (see configs/default.yaml).")
    p.add_argument(
        "--run_dir",
        type=Path,
        default=None,
        help="Optional explicit run directory. If not set, one is created.",
    )
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="auto | none | /path/to/checkpoint.pt (overrides config.train.resume)",
    )
    p.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config key-values, e.g. train.batch_size=8",
    )
    return p.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    cfg = yaml.safe_load(path.read_text())
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a YAML mapping.")
    return cfg


def set_by_dotted_key(cfg: Dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    cur: Dict[str, Any] = cfg
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def apply_overrides(cfg: Dict[str, Any], overrides: list[str]) -> Dict[str, Any]:
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"Bad override (expected key=value): {ov}")
        key, raw = ov.split("=", 1)

        # Minimal type inference (good enough for configs)
        if raw.lower() in {"true", "false"}:
            val: Any = raw.lower() == "true"
        else:
            try:
                val = int(raw)
            except ValueError:
                try:
                    val = float(raw)
                except ValueError:
                    val = raw

        set_by_dotted_key(cfg, key, val)
    return cfg


def resolve_run_dir(cfg: Dict[str, Any], explicit_run_dir: Optional[Path], dist: DistInfo) -> Path:
    if explicit_run_dir is not None:
        run_dir = explicit_run_dir
    else:
        base = Path(cfg["train"]["run_dir"])
        project = cfg["project"]["name"]
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = base / f"{project}_{stamp}"

    if is_main_process(dist):
        run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_dataloaders(cfg: Dict[str, Any], dist: DistInfo):
    train_dir = Path(cfg["data"]["processed_dir"])
    casp_root = train_dir.parent
    val_dir = casp_root / "validation"

    max_len = cfg["data"].get("max_length", None)

    train_ds = ProteinNetMemmapDataset(train_dir, max_length=max_len)
    val_ds = ProteinNetMemmapDataset(val_dir, max_length=max_len) if val_dir.exists() else None

    if dist.enabled:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds, shuffle=True)
        val_sampler = (
            torch.utils.data.distributed.DistributedSampler(val_ds, shuffle=False) if val_ds is not None else None
        )
        shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=int(cfg["train"]["num_workers"]),
        pin_memory=True,
        collate_fn=collate_batch,
        persistent_workers=int(cfg["train"]["num_workers"]) > 0,
    )

    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=int(cfg["train"]["batch_size"]),
            shuffle=False,
            sampler=val_sampler,
            num_workers=int(cfg["train"]["num_workers"]),
            pin_memory=True,
            collate_fn=collate_batch,
            persistent_workers=int(cfg["train"]["num_workers"]) > 0,
        )

    return train_loader, val_loader, train_sampler, val_sampler


def build_model(cfg: Dict[str, Any]) -> ProteinTransformer:
    m = cfg["model"]
    data_max_len = cfg["data"].get("max_length", 4096) or 4096
    model_cfg = ProteinTransformerConfig(
        vocab_size=int(m["vocab_size"]),
        evo_dim=int(m["evo_dim"]),
        d_model=int(m["d_model"]),
        n_heads=int(m["n_heads"]),
        n_layers=int(m["n_layers"]),
        dim_feedforward=int(m["dim_feedforward"]),
        dropout=float(m["dropout"]),
        max_len=int(data_max_len),
    )
    return ProteinTransformer(model_cfg)


def make_scheduler(cfg: Dict[str, Any], optimizer: torch.optim.Optimizer, total_steps: int):
    s = cfg.get("scheduler", {})
    name = s.get("name", "cosine")
    base_lr = float(cfg["train"]["lr"])
    warmup = int(s.get("warmup_steps", 0))
    min_lr = float(s.get("min_lr", 0.0))

    if name != "cosine":
        raise ValueError(f"Unsupported scheduler: {name}")

    def lr_lambda(step: int) -> float:
        if total_steps <= 0:
            return 1.0
        if warmup > 0 and step < warmup:
            return max(1e-8, step / float(warmup))

        # cosine decay after warmup
        t = (step - warmup) / float(max(1, total_steps - warmup))
        t = min(1.0, max(0.0, t))
        cosine = 0.5 * (1.0 + math.cos(math.pi * t))
        lr = min_lr + (base_lr - min_lr) * cosine
        return lr / base_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def reduce_mean(x: torch.Tensor, dist: DistInfo) -> torch.Tensor:
    if not dist.enabled:
        return x
    x = x.clone()
    torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.SUM)
    x /= dist.world_size
    return x


def _masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """SmoothL1 over valid points only."""
    loss = torch.nn.functional.smooth_l1_loss(pred, target, reduction="none")  # (B,L,3)
    loss = loss.sum(dim=-1)  # (B,L)
    loss = loss[mask]
    return loss.mean()


def kabsch_align_batch(P: torch.Tensor, Q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Align P onto Q per-sample (batched), using a weighted Kabsch with a padding mask.

    Args:
        P: (B,L,3) float32
        Q: (B,L,3) float32
        mask: (B,L) bool

    Returns:
        P_aligned: (B,L,3)
    """
    if P.dtype != torch.float32:
        P = P.float()
    if Q.dtype != torch.float32:
        Q = Q.float()

    w = mask.to(dtype=P.dtype).unsqueeze(-1)  # (B,L,1)
    denom = w.sum(dim=1, keepdim=True).clamp_min(1.0)  # (B,1,1)

    P_mean = (P * w).sum(dim=1, keepdim=True) / denom
    Q_mean = (Q * w).sum(dim=1, keepdim=True) / denom

    Pc = (P - P_mean) * w
    Qc = (Q - Q_mean) * w

    C = torch.bmm(Pc.transpose(1, 2), Qc)  # (B,3,3)

    # SVD on each 3x3 covariance
    V, _, Wt = torch.linalg.svd(C, full_matrices=False)

    # Reflection fix: enforce det(R)=+1
    det = torch.det(torch.bmm(V, Wt))  # (B,)
    d = torch.sign(det)
    d = torch.where(d == 0, torch.ones_like(d), d)

    D = torch.eye(3, device=P.device, dtype=P.dtype).expand(P.shape[0], 3, 3).clone()
    D[:, 2, 2] = d

    R = torch.bmm(torch.bmm(V, D), Wt)  # (B,3,3)
    P_aligned = torch.bmm(P - P_mean, R) + Q_mean
    return P_aligned


def compute_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, *, align: bool) -> torch.Tensor:
    """Main training loss.

    If align=True, we Kabsch-align pred to target per protein before SmoothL1.
    """
    pred_f = pred.float()
    target_f = target.float()

    if align:
        pred_f = kabsch_align_batch(pred_f, target_f, mask)

    return _masked_smooth_l1(pred_f, target_f, mask)


def apply_evo_dropout(evo: torch.Tensor, p: float) -> torch.Tensor:
    """Drop evolutionary features for a subset of proteins (per-sample).

    This trains a single checkpoint that can work with or without evo features.
    """
    if p <= 0.0:
        return evo
    if p >= 1.0:
        return torch.zeros_like(evo)

    B = evo.shape[0]
    drop = (torch.rand(B, device=evo.device) < p).view(B, 1, 1)
    return evo.masked_fill(drop, 0.0)


def main() -> None:
    args = parse_args()
    dist = init_distributed()

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.override)

    if args.resume is not None:
        cfg.setdefault("train", {})
        cfg["train"]["resume"] = args.resume

    seed_everything(int(cfg["train"]["seed"]))

    device = torch.device("cuda", dist.local_rank) if torch.cuda.is_available() else torch.device("cpu")

    # A100: TF32 is a nice throughput win and usually fine for this task.
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ---- run dirs ----
    run_dir = resolve_run_dir(cfg, args.run_dir, dist)
    ckpt_dir = run_dir / "checkpoints"
    tb_dir = run_dir / "tb"
    metrics_csv = run_dir / "metrics.csv"
    loss_png = run_dir / "loss_curve.png"

    if is_main_process(dist):
        (run_dir / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        print(f"Run dir: {run_dir}")
        print(f"Device: {device} | torch={torch.__version__} | cuda={torch.version.cuda}")

    # ---- data ----
    train_loader, val_loader, train_sampler, _ = build_dataloaders(cfg, dist)

    # ---- model ----
    model = build_model(cfg).to(device)
    if dist.enabled:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.local_rank],
            output_device=dist.local_rank,
            find_unused_parameters=False,
        )

    optimizer = torch.optim.AdamW(
        unwrap_ddp(model).parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
        betas=(0.9, 0.98),
    )

    grad_accum = int(cfg["train"].get("gradient_accumulation", 1))
    steps_per_epoch = math.ceil(len(train_loader) / max(1, grad_accum))
    total_steps = int(cfg["train"]["epochs"]) * steps_per_epoch

    scheduler = make_scheduler(cfg, optimizer, total_steps=total_steps)

    amp_enabled = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    align_loss = bool(cfg["train"].get("align_loss", True))
    evo_drop_p = float(cfg["train"].get("evo_dropout_prob", 0.0))

    # ---- logging ----
    writer = SummaryWriter(log_dir=str(tb_dir)) if is_main_process(dist) else None

    # ---- resume ----
    start_epoch = 0
    global_step = 0
    best_val = float("inf")

    resume_mode = str(cfg["train"].get("resume", "auto")).lower()
    ckpt_path: Optional[Path]
    if resume_mode == "auto":
        ckpt_path = find_latest_checkpoint(ckpt_dir)
    elif resume_mode in {"none", "false", "0"}:
        ckpt_path = None
    else:
        ckpt_path = Path(resume_mode)

    if ckpt_path is not None and ckpt_path.exists():
        payload = load_checkpoint(
            ckpt_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location="cpu",
        )
        start_epoch = int(payload.get("epoch", -1)) + 1
        global_step = int(payload.get("global_step", 0))
        best_val = float(payload.get("extra", {}).get("best_val", best_val))
        if is_main_process(dist):
            print(f"Resumed from {ckpt_path} (epoch={start_epoch}, step={global_step})")

    # CSV header
    if is_main_process(dist) and (not metrics_csv.exists()):
        with metrics_csv.open("w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "val_rmsd"])

    epochs = int(cfg["train"]["epochs"])
    log_every = int(cfg["train"].get("log_every_steps", 50))
    val_every = int(cfg["train"].get("validate_every_epochs", 1))
    save_every = int(cfg["train"].get("save_every_steps", 500))
    clip_norm = float(cfg["train"].get("grad_clip_norm", 1.0))
    keep_last_k = int(cfg["train"].get("keep_last_k", 3))

    if is_main_process(dist):
        print(f"align_loss={align_loss} | evo_dropout_prob={evo_drop_p} | amp={amp_enabled} | grad_accum={grad_accum}")

    # Training loop
    for epoch in range(start_epoch, epochs):
        if dist.enabled and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        optimizer.zero_grad(set_to_none=True)

        running_loss = 0.0
        n_batches = 0

        t_epoch0 = time.perf_counter()
        t_step0 = time.perf_counter()

        pbar = tqdm(train_loader, disable=not is_main_process(dist), desc=f"Epoch {epoch+1}/{epochs}")

        for step, batch in enumerate(pbar):
            tokens = batch["tokens"].to(device, non_blocking=True)
            evo = batch["evo"].to(device, non_blocking=True)
            target = batch["ca_coords"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            pad_mask = batch["pad_mask"].to(device, non_blocking=True)

            # Hybrid trick: sometimes drop evo so the model learns to operate without it.
            if evo_drop_p > 0.0:
                evo = apply_evo_dropout(evo, evo_drop_p)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                pred = model(tokens=tokens, evo=evo, pad_mask=pad_mask)

            # Important: compute the loss in float32 (SVD/align wants fp32, and is more stable).
            with torch.cuda.amp.autocast(enabled=False):
                loss = compute_loss(pred, target, mask, align=align_loss)
                loss_to_backprop = loss / max(1, grad_accum)

            scaler.scale(loss_to_backprop).backward()

            # optimizer step (after grad accumulation)
            if (step + 1) % grad_accum == 0:
                if clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(unwrap_ddp(model).parameters(), max_norm=clip_norm)

                stepped = True
                if amp_enabled:
                    scale_before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    stepped = scaler.get_scale() >= scale_before
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)

                if stepped:
                    scheduler.step()

                global_step += 1

                # scalar logs
                if writer is not None and (global_step % log_every == 0):
                    dt_s = max(1e-6, time.perf_counter() - t_step0)
                    it_per_s = log_every / dt_s

                    writer.add_scalar("train/loss_step", float(loss.item()), global_step)
                    writer.add_scalar("train/lr", float(optimizer.param_groups[0]["lr"]), global_step)
                    writer.add_scalar("train/it_per_s", float(it_per_s), global_step)

                    t_step0 = time.perf_counter()

                # periodic checkpoint
                if is_main_process(dist) and (save_every > 0) and (global_step % save_every == 0):
                    save_checkpoint(
                        ckpt_path=ckpt_dir / "latest.pt",
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch=epoch,
                        global_step=global_step,
                        config=cfg,
                        extra={"best_val": best_val},
                    )

            running_loss += float(loss.item())
            n_batches += 1
            pbar.set_postfix(loss=running_loss / max(1, n_batches))

        # ---- epoch summary ----
        train_loss = torch.tensor(running_loss / max(1, n_batches), device=device)
        train_loss = reduce_mean(train_loss, dist).item()

        val_loss = float("nan")
        val_rmsd = float("nan")

        do_val = (val_loader is not None) and ((epoch + 1) % val_every == 0)
        if do_val:
            model.eval()
            v_loss_sum = 0.0
            v_batches = 0
            rmsd_sum = 0.0
            rmsd_batches = 0

            with torch.no_grad():
                for vbatch in tqdm(val_loader, disable=not is_main_process(dist), desc="Validating"):
                    tokens = vbatch["tokens"].to(device, non_blocking=True)
                    evo = vbatch["evo"].to(device, non_blocking=True)
                    target = vbatch["ca_coords"].to(device, non_blocking=True)
                    mask = vbatch["mask"].to(device, non_blocking=True)
                    pad_mask = vbatch["pad_mask"].to(device, non_blocking=True)

                    with torch.cuda.amp.autocast(enabled=amp_enabled):
                        pred = model(tokens=tokens, evo=evo, pad_mask=pad_mask)

                    with torch.cuda.amp.autocast(enabled=False):
                        vloss = compute_loss(pred, target, mask, align=align_loss)

                        # RMSD (aligned) in float32 for safety
                        br = batch_rmsd(pred.float(), target.float(), mask, align=True)

                    v_loss_sum += float(vloss.item())
                    v_batches += 1

                    if not torch.isnan(br):
                        rmsd_sum += float(br.item())
                        rmsd_batches += 1

            val_loss_t = torch.tensor(v_loss_sum / max(1, v_batches), device=device)
            val_loss = reduce_mean(val_loss_t, dist).item()

            val_rmsd_t = torch.tensor(rmsd_sum / max(1, rmsd_batches), device=device)
            val_rmsd = reduce_mean(val_rmsd_t, dist).item()

            if writer is not None:
                writer.add_scalar("val/loss", val_loss, epoch)
                writer.add_scalar("val/rmsd", val_rmsd, epoch)

            if val_loss < best_val:
                best_val = val_loss
                if is_main_process(dist):
                    save_checkpoint(
                        ckpt_path=ckpt_dir / "best.pt",
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch=epoch,
                        global_step=global_step,
                        config=cfg,
                        extra={"best_val": best_val},
                    )

        # Always save latest + epoch checkpoint
        if is_main_process(dist):
            save_checkpoint(
                ckpt_path=ckpt_dir / "latest.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                config=cfg,
                extra={"best_val": best_val},
            )
            save_checkpoint(
                ckpt_path=ckpt_dir / f"epoch_{epoch:03d}.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                config=cfg,
                extra={"best_val": best_val},
            )
            rotate_checkpoints(ckpt_dir, keep_last_k=keep_last_k)

            with metrics_csv.open("a", newline="") as f:
                csv.writer(f).writerow([epoch, train_loss, val_loss, val_rmsd])

            plot_losses(metrics_csv, loss_png, title="ProteinNet hybrid training")

            t_epoch = time.perf_counter() - t_epoch0
            print(
                f"[epoch {epoch:03d}] train_loss={train_loss:.5f} val_loss={val_loss:.5f} "
                f"val_rmsd={val_rmsd:.3f} best_val={best_val:.5f} (epoch_time={t_epoch/60.0:.1f} min)"
            )

    if writer is not None:
        writer.close()

    cleanup_distributed(dist)


if __name__ == "__main__":
    main()