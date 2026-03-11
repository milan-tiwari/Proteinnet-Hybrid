#!/usr/bin/env python
"""
Export ProteinNet CA-trace predictions to PDB for *actual structure visualization*.

What this does:
- Loads your trained checkpoint (best.pt or latest.pt)
- Runs inference on a few samples (default: from validation split)
- Writes:
    <out_dir>/
      <id>_true_ca.pdb
      <id>_pred_ca.pdb
      manifest.csv
      view_in_pymol_local.pml

Key fixes vs the old script:
- Works no matter where you run it from (fixes "No module named src")
- Sanitizes IDs (fixes '#' and weird PyMOL object names)
- Writes a *local-friendly* PyMOL script that uses relative filenames
- Uses PyMOL representations that make CA-only structures readable (trace/cartoon)
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from tqdm import tqdm

# --- Make `import src...` work when running `python scripts/...` ---
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.collate import collate_batch
from src.data.memmap_dataset import ProteinNetMemmapDataset
from src.data.proteinnet_parser import AMINO_ACIDS, PAD_ID, UNK_ID
from src.models.transformer import ProteinTransformer, ProteinTransformerConfig
from src.utils.checkpointing import load_checkpoint
from src.utils.metrics import rmsd


AA3 = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE",
    "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU",
    "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG",
    "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
}
ID_TO_AA = {i + 1: aa for i, aa in enumerate(AMINO_ACIDS)}
ID_TO_AA[UNK_ID] = "X"
ID_TO_AA[PAD_ID] = "-"


def sanitize_id(s: str) -> str:
    """Make an ID safe for filenames + PyMOL object names."""
    s = s.strip()
    s = s.replace("/", "_").replace("#", "_")
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    s = s.strip("_")
    if not s:
        s = "sample"
    if s[0].isdigit():
        s = f"p{s}"
    return s


def tokens_to_resnames(tokens: np.ndarray) -> List[str]:
    out: List[str] = []
    for t in tokens.tolist():
        aa = ID_TO_AA.get(int(t), "X")
        out.append(AA3.get(aa, "UNK"))
    return out


def write_ca_pdb(
    out_path: Path,
    coords: np.ndarray,         # (N,3) for valid residues only
    resnames: List[str],        # (N,)
    resi_numbers: np.ndarray,   # (N,) original residue indices (1-based)
    chain_id: str = "A",
    b_factors: Optional[np.ndarray] = None,  # (N,) optional per-residue values
) -> None:
    """
    Write a CA-only PDB with reasonable formatting.

    Note:
      - CA-only means "sticks" can look weird if PyMOL guesses bonds.
      - Our PyMOL script uses TRACE/CARTOON to avoid that problem.
    """
    if b_factors is None:
        b_factors = np.zeros((coords.shape[0],), dtype=np.float32)

    lines: List[str] = []
    atom_serial = 1

    for xyz, resn, resi, b in zip(coords, resnames, resi_numbers, b_factors):
        x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
        b = float(b)
        # PDB-ish ATOM line; PyMOL is forgiving.
        lines.append(
            f"ATOM  {atom_serial:5d}  CA  {resn:>3s} {chain_id}{int(resi):4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{b:6.2f}           C"
        )
        atom_serial += 1

    lines.append("END")
    out_path.write_text("\n".join(lines) + "\n")


def build_model_from_cfg(cfg: Dict) -> ProteinTransformer:
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


def pick_indices(
    ds: ProteinNetMemmapDataset,
    num_samples: int,
    *,
    mode: str,
    start_idx: int,
    seed: int,
    min_valid: int,
) -> List[int]:
    rng = random.Random(seed)

    def valid_count(i: int) -> int:
        s = ds[i]
        return int(np.asarray(s["mask"], dtype=bool).sum())

    if mode == "first":
        out: List[int] = []
        i = start_idx
        while i < len(ds) and len(out) < num_samples:
            if valid_count(i) >= min_valid:
                out.append(i)
            i += 1
        return out

    if mode == "random":
        candidates = list(range(len(ds)))
        rng.shuffle(candidates)
        out = []
        for i in candidates:
            if valid_count(i) >= min_valid:
                out.append(i)
            if len(out) >= num_samples:
                break
        return out

    raise ValueError(f"Unknown mode: {mode}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export ProteinNet predictions as PDB for PyMOL visualization.")
    p.add_argument("--config", type=Path, required=True, help="Config used for training (prefer run_dir/config_resolved.yaml).")
    p.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint path (best.pt or latest.pt).")
    p.add_argument("--split_dir", type=Path, default=None, help="Processed split dir (default: validation next to train dir).")
    p.add_argument("--out_dir", type=Path, default=Path("exports"), help="Output directory for PDB files.")
    p.add_argument("--num_samples", type=int, default=10, help="How many proteins to export.")
    p.add_argument("--mode", type=str, default="first", choices=["first", "random"], help="How to select proteins.")
    p.add_argument("--start_idx", type=int, default=0, help="Start index for mode=first.")
    p.add_argument("--seed", type=int, default=7, help="Seed for mode=random.")
    p.add_argument("--min_valid", type=int, default=50, help="Skip proteins with fewer than this many valid residues.")
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    # Resolve processed split
    train_dir = Path(cfg["data"]["processed_dir"])
    casp_root = train_dir.parent
    default_val = casp_root / "validation"
    split_dir = args.split_dir or default_val

    if not split_dir.exists():
        raise FileNotFoundError(
            f"Processed split_dir not found: {split_dir}\n"
            f"Tip: pass --split_dir explicitly, e.g. --split_dir /scratch/$USER/proteinnet/data/processed/casp12/validation"
        )

    ds = ProteinNetMemmapDataset(split_dir, max_length=cfg["data"].get("max_length", None))
    indices = pick_indices(
        ds, args.num_samples,
        mode=args.mode,
        start_idx=args.start_idx,
        seed=args.seed,
        min_valid=args.min_valid,
    )

    if not indices:
        raise RuntimeError(
            f"No samples found with min_valid={args.min_valid}. "
            f"Try --min_valid 0 or --mode random."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_cfg(cfg).to(device)
    load_checkpoint(args.checkpoint, model=model, map_location="cpu")
    model.eval()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.out_dir / "manifest.csv"
    with manifest_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "raw_id", "safe_id", "valid_res", "rmsd_aligned_A", "true_pdb", "pred_pdb"])

        print(f"Exporting {len(indices)} proteins from: {split_dir}")
        for idx in tqdm(indices, desc="Export", unit="protein"):
            sample = ds[idx]
            batch = collate_batch([sample])

            tokens = batch["tokens"].to(device)
            evo = batch["evo"].to(device)
            target = batch["ca_coords"].to(device)
            mask = batch["mask"].to(device)
            pad_mask = batch["pad_mask"].to(device)

            pred = model(tokens=tokens, evo=evo, pad_mask=pad_mask)

            L = int(batch["lengths"][0])
            valid = mask[0, :L].bool()

            true_ca = target[0, :L][valid].float().cpu()  # (N,3)
            pred_ca = pred[0, :L][valid].float().cpu()    # (N,3)

            if true_ca.shape[0] < 3:
                continue

            r = float(rmsd(pred_ca, true_ca, align=True).item())

            raw_id = str(batch["ids"][0])
            safe_id = sanitize_id(raw_id)

            # B-factor: per-residue CA error *after alignment*, helps color in PyMOL if desired.
            # We align by calling rmsd(..., align=True) above, but for b-factors we do a fresh alignment.
            # This is cheap for N residues.
            # (If alignment ever fails, just fall back to zeros.)
            b_factors = np.zeros((true_ca.shape[0],), dtype=np.float32)
            try:
                # Kabsch alignment (local copy)
                P = pred_ca.clone()
                Q = true_ca.clone()
                P_mean = P.mean(dim=0, keepdim=True)
                Q_mean = Q.mean(dim=0, keepdim=True)
                Pc = P - P_mean
                Qc = Q - Q_mean
                C = Pc.T @ Qc
                V, S, Wt = torch.linalg.svd(C, full_matrices=False)
                d = torch.sign(torch.det(V @ Wt))
                D = torch.diag(torch.tensor([1.0, 1.0, float(d)], dtype=P.dtype))
                Rm = V @ D @ Wt
                P_aligned = (Pc @ Rm) + Q_mean
                b_factors = torch.linalg.norm(P_aligned - Q, dim=-1).numpy().astype(np.float32)
            except Exception:
                pass

            resnames = tokens_to_resnames(batch["tokens"][0, :L][valid].cpu().numpy())
            resi_numbers = (torch.nonzero(valid, as_tuple=False).squeeze(-1).cpu().numpy() + 1).astype(np.int32)

            true_pdb = args.out_dir / f"{safe_id}_true_ca.pdb"
            pred_pdb = args.out_dir / f"{safe_id}_pred_ca.pdb"

            write_ca_pdb(true_pdb, true_ca.numpy(), resnames, resi_numbers, b_factors=None)
            write_ca_pdb(pred_pdb, pred_ca.numpy(), resnames, resi_numbers, b_factors=b_factors)

            w.writerow([idx, raw_id, safe_id, int(valid.sum().item()), f"{r:.4f}", true_pdb.name, pred_pdb.name])

            print(f"[{idx}] {raw_id}  valid={int(valid.sum())}  RMSD(aligned)={r:.3f} Å")
            print(f"  -> {true_pdb}")
            print(f"  -> {pred_pdb}")

    # --- Write a LOCAL-FRIENDLY PyMOL script (relative filenames only) ---
    pml = args.out_dir / "view_in_pymol_local.pml"
    pml_lines = [
        "# Run this from inside the exports folder (or `cd` there in PyMOL)",
        "bg_color white",
        "set ray_opaque_background, off",
        "",
        "# IMPORTANT: for CA-only PDBs, TRACE/CARTOON is cleaner than STICKS.",
        "hide everything",
        "set cartoon_sampling, 10",
        "set cartoon_trace_atoms, 1",
        "",
    ]

    # Group objects so your screen doesn't become a spaghetti explosion.
    rows = list(csv.DictReader(manifest_path.read_text().splitlines()))
    for row in rows:
        sid = row["safe_id"]
        true_file = row["true_pdb"]
        pred_file = row["pred_pdb"]

        obj_true = f"prot_{sid}_true"
        obj_pred = f"prot_{sid}_pred"
        grp = f"grp_{sid}"

        pml_lines += [
            f"load {true_file}, {obj_true}",
            f"load {pred_file}, {obj_pred}",
            f"group {grp}, {obj_true} {obj_pred}",
            f"disable {grp}",  # keep disabled by default
            f"color green, {obj_true}",
            f"color magenta, {obj_pred}",
            f"show trace, {obj_true}",
            f"show trace, {obj_pred}",
            f"set trace_radius, 0.4, {grp}",
            f"align {obj_pred}, {obj_true}",
            "",
        ]

    if rows:
        first = rows[0]["safe_id"]
        pml_lines += [
            f"enable grp_{first}",
            f"zoom grp_{first}, 2",
            "",
            "# Tip: enable another group with e.g. `enable grp_<id>`",
        ]

    pml.write_text("\n".join(pml_lines) + "\n")
    print(f"\nWrote: {manifest_path}")
    print(f"Wrote: {pml}")
    print("\nNext step (PyMOL local): copy the whole out_dir to your laptop and run `@view_in_pymol_local.pml`.")


if __name__ == "__main__":
    main()