#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from huggingface_hub import hf_hub_download

# Make `src...` imports work
REPO_ROOT = Path(__file__).resolve().parent
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.transformer import ProteinTransformer, ProteinTransformerConfig

# --- Tokenization (must match training) ---
# ProteinNet training used vocab_size=22 and token IDs where 0=PAD, 1..20=AA, 21=UNK (common scheme).
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_ID = {aa: i + 1 for i, aa in enumerate(AA_LIST)}
PAD_ID = 0
UNK_ID = 21

ID_TO_AA3 = {
    "A":"ALA","C":"CYS","D":"ASP","E":"GLU","F":"PHE","G":"GLY","H":"HIS","I":"ILE","K":"LYS","L":"LEU",
    "M":"MET","N":"ASN","P":"PRO","Q":"GLN","R":"ARG","S":"SER","T":"THR","V":"VAL","W":"TRP","Y":"TYR",
}
def aa3(aa: str) -> str:
    return ID_TO_AA3.get(aa, "UNK")

def sanitize_id(s: str) -> str:
    s = s.strip()
    s = s.replace("#","_").replace("/","_")
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    s = s.strip("_")
    if not s:
        s = "seq"
    if s[0].isdigit():
        s = f"p{s}"
    return s

def read_fasta(path: Path) -> List[Tuple[str,str]]:
    items = []
    cur_id = None
    cur_seq = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if cur_id is not None:
                items.append((sanitize_id(cur_id), "".join(cur_seq).upper()))
            cur_id = line[1:].strip() or "seq"
            cur_seq = []
        else:
            cur_seq.append(line)
    if cur_id is not None:
        items.append((sanitize_id(cur_id), "".join(cur_seq).upper()))
    if not items:
        raise ValueError(f"No FASTA records found in {path}")
    return items

# --- A3M/MSA -> simple 21-d evo features (freqs + info content) ---
AA_INDEX = {aa:i for i, aa in enumerate(AA_LIST)}

def read_a3m(path: Path) -> List[str]:
    seqs = []
    cur = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if cur:
                seqs.append("".join(cur))
            cur = []
        else:
            # A3M uses lowercase insertions; remove lowercase chars
            line = "".join([c for c in line if not c.islower()])
            cur.append(line)
    if cur:
        seqs.append("".join(cur))
    if not seqs:
        raise ValueError(f"No sequences found in A3M: {path}")
    return seqs

def build_evo_from_msa(msa: List[str], L: int) -> np.ndarray:
    # 21 dims: 20 AA freqs + 1 "information content" scalar
    counts = np.zeros((L, 20), dtype=np.float32)
    total = np.zeros((L,), dtype=np.float32)

    for s in msa:
        if len(s) < L:
            continue
        for i, c in enumerate(s[:L]):
            if c == "-" or c == ".":
                continue
            j = AA_INDEX.get(c)
            if j is None:
                continue
            counts[i, j] += 1.0
            total[i] += 1.0

    freqs = counts / np.clip(total[:, None], 1.0, None)
    # info content ~ sum(p * log(p)) (negative entropy), scaled
    eps = 1e-8
    entropy = -(freqs * np.log(freqs + eps)).sum(axis=1)  # [0..~3]
    info = (np.log(20.0) - entropy).astype(np.float32)    # higher = more conserved
    evo = np.concatenate([freqs, info[:, None]], axis=1)  # (L,21)
    return evo

def tokenize(seq: str, max_len: Optional[int]) -> np.ndarray:
    seq = seq.strip().upper()
    if max_len is not None:
        seq = seq[:max_len]
    toks = np.full((len(seq),), UNK_ID, dtype=np.int64)
    for i, c in enumerate(seq):
        toks[i] = AA_TO_ID.get(c, UNK_ID)
    return toks

def write_ca_pdb(path: Path, coords: np.ndarray, seq: str, chain="A"):
    # coords: (L,3)
    lines = []
    atom_i = 1
    for i, (x,y,z) in enumerate(coords.tolist(), start=1):
        resn = aa3(seq[i-1]) if i-1 < len(seq) else "UNK"
        lines.append(
            f"ATOM  {atom_i:5d}  CA  {resn:>3s} {chain}{i:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C"
        )
        atom_i += 1
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")

def load_cfg_and_model(cfg_path: Path, ckpt_path: Path, device: torch.device):
    cfg = yaml.safe_load(cfg_path.read_text())
    m = cfg["model"]
    max_len = int(cfg["data"].get("max_length", 1024))

    model_cfg = ProteinTransformerConfig(
        vocab_size=int(m["vocab_size"]),
        evo_dim=int(m["evo_dim"]),
        d_model=int(m["d_model"]),
        n_heads=int(m["n_heads"]),
        n_layers=int(m["n_layers"]),
        dim_feedforward=int(m["dim_feedforward"]),
        dropout=float(m["dropout"]),
        max_len=max_len,
    )
    model = ProteinTransformer(model_cfg).to(device)
    payload = torch.load(ckpt_path, map_location="cpu")
    state = payload.get("model", payload)
    model.load_state_dict(state, strict=True)
    model.eval()
    return cfg, model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--mode", choices=["lite","msa"], default="lite")
    ap.add_argument("--a3m", type=Path, default=None, help="Required for --mode msa")
    ap.add_argument("--device", choices=["cpu","cuda","auto"], default="auto")
    ap.add_argument("--dtype", choices=["fp32","fp16"], default="fp32")

    # weights: either local checkpoint OR HF
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=None, help="config_resolved.yaml (if checkpoint doesn't include it)")
    ap.add_argument("--hf_repo", type=str, default=None)
    ap.add_argument("--hf_filename", type=str, default="best.pt")

    ap.add_argument("--max_len", type=int, default=1024)

    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve checkpoint path
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        if not args.hf_repo:
            raise SystemExit("Provide --checkpoint or (--hf_repo and optional --hf_filename).")
        ckpt_path = Path(hf_hub_download(repo_id=args.hf_repo, filename=args.hf_filename))
    ckpt_path = ckpt_path.resolve()

    # Resolve config: prefer payload["config"] if exists; else require --config
    payload = torch.load(ckpt_path, map_location="cpu")
    cfg = payload.get("config", None)
    if cfg is None:
        if args.config is None:
            raise SystemExit("Checkpoint has no embedded config. Pass --config /path/to/config_resolved.yaml")
        cfg_path = args.config
    else:
        # write temp config for model construction
        cfg_path = args.out_dir / "_config_tmp.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    use_fp16 = (args.dtype == "fp16" and device.type == "cuda")

    cfg_loaded, model = load_cfg_and_model(cfg_path, ckpt_path, device)
    max_len = min(int(cfg_loaded["data"].get("max_length", args.max_len)), args.max_len)

    records = read_fasta(args.fasta)
    msa = None
    if args.mode == "msa":
        if args.a3m is None:
            raise SystemExit("--mode msa requires --a3m file.a3m")
        msa = read_a3m(args.a3m)

    manifest = []
    for rid, seq in records:
        toks = tokenize(seq, max_len=max_len)
        L = toks.shape[0]

        if args.mode == "msa":
            evo = build_evo_from_msa(msa, L)
        else:
            evo = np.zeros((L, 21), dtype=np.float32)

        # batchify to (B,L)
        tokens_t = torch.from_numpy(toks[None, :]).to(device)
        evo_t = torch.from_numpy(evo[None, :, :]).to(device)
        pad_mask = torch.zeros((1, L), dtype=torch.bool, device=device)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=use_fp16):
                pred = model(tokens=tokens_t, evo=evo_t, pad_mask=pad_mask)  # (1,L,3)

        pred_ca = pred[0].float().cpu().numpy()
        out_pdb = args.out_dir / f"{rid}_pred_ca.pdb"
        write_ca_pdb(out_pdb, pred_ca, seq[:L])

        meta = {
            "id": rid,
            "length": int(L),
            "mode": args.mode,
            "device": str(device),
            "dtype": "fp16" if use_fp16 else "fp32",
            "checkpoint": str(ckpt_path),
            "pdb": str(out_pdb.name),
        }
        (args.out_dir / f"{rid}_meta.json").write_text(json.dumps(meta, indent=2))
        manifest.append(meta)
        print(f"[OK] {rid} -> {out_pdb}")

    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nDone. Wrote {len(manifest)} structures to: {args.out_dir}")

if __name__ == "__main__":
    main()
