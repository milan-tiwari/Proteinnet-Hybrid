#!/usr/bin/env python3
"""
dataset_setup.py

One-shot script to download + extract ProteinNet (human_readable) and run preprocessing
into memory-mapped arrays used by training.

Designed for headless use on SOL (no browser/Jupyter needed).

Example:
  python dataset_setup.py --casp casp12 --train_threshold 90 --max_length 1024 \
    --work_dir /scratch/$USER/proteinnet
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

DEFAULT_URLS = {
    "casp7":  "https://sharehost.hms.harvard.edu/sysbio/alquraishi/proteinnet/human_readable/casp7.tar.gz",
    "casp8":  "https://sharehost.hms.harvard.edu/sysbio/alquraishi/proteinnet/human_readable/casp8.tar.gz",
    "casp9":  "https://sharehost.hms.harvard.edu/sysbio/alquraishi/proteinnet/human_readable/casp9.tar.gz",
    "casp10": "https://sharehost.hms.harvard.edu/sysbio/alquraishi/proteinnet/human_readable/casp10.tar.gz",
    "casp11": "https://sharehost.hms.harvard.edu/sysbio/alquraishi/proteinnet/human_readable/casp11.tar.gz",
    "casp12": "https://sharehost.hms.harvard.edu/sysbio/alquraishi/proteinnet/human_readable/casp12.tar.gz",
}

def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)

def which_any(candidates: list[str]) -> str | None:
    for c in candidates:
        p = shutil.which(c)
        if p:
            return p
    return None

def download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"Found existing archive: {out_path} ({out_path.stat().st_size/1e9:.2f} GB). Skipping download.")
        return

    wget = which_any(["wget"])
    curl = which_any(["curl"])
    if wget:
        run([wget, "-c", url, "-O", str(out_path)])
    elif curl:
        run([curl, "-L", "-C", "-", url, "-o", str(out_path)])
    else:
        raise RuntimeError("Neither wget nor curl is available. Load a module that provides one.")

def safe_extract_tar_gz(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {archive} -> {dest}")
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(path=dest)

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--casp", default="casp12", choices=sorted(DEFAULT_URLS.keys()))
    ap.add_argument("--url", default=None, help="Override download URL (optional).")
    ap.add_argument("--work_dir", required=True, help="Base directory (recommend /scratch/$USER/proteinnet)")
    ap.add_argument("--train_threshold", type=int, default=90, choices=[30,50,70,90,95,100])
    ap.add_argument("--max_length", type=int, default=1024, help="Truncate proteins longer than this.")
    ap.add_argument("--repo_root", default=".", help="Path to repo root (contains scripts/preprocess_proteinnet.py)")
    ap.add_argument("--skip_download", action="store_true")
    ap.add_argument("--skip_extract", action="store_true")
    ap.add_argument("--skip_preprocess", action="store_true")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    preprocess_script = repo_root / "scripts" / "preprocess_proteinnet.py"
    if not preprocess_script.exists():
        print(f"ERROR: {preprocess_script} not found. Run from repo root or pass --repo_root.", file=sys.stderr)
        return 2

    work_dir = Path(os.path.expandvars(args.work_dir)).resolve()
    raw_root = work_dir / "data" / "raw"
    out_root = work_dir / "data" / "processed"
    casp_dir = raw_root / args.casp

    url = args.url or DEFAULT_URLS[args.casp]
    archive = raw_root / f"{args.casp}.tar.gz"

    print("=== ProteinNet setup ===")
    print(f"CASP:            {args.casp}")
    print(f"URL:             {url}")
    print(f"Work dir:        {work_dir}")
    print(f"Raw root:        {raw_root}")
    print(f"Processed root:  {out_root}")
    print(f"Train thinning:  training_{args.train_threshold}")
    print(f"Max length:      {args.max_length}")
    print(f"Repo root:       {repo_root}")

    if not args.skip_download:
        download(url, archive)

    if not args.skip_extract:
        if casp_dir.exists() and any(casp_dir.iterdir()):
            print(f"Found existing extracted directory: {casp_dir}. Skipping extract.")
        else:
            safe_extract_tar_gz(archive, raw_root)

    if not args.skip_preprocess:
        out_root.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, str(preprocess_script),
            "--casp", args.casp,
            "--raw_root", str(raw_root),
            "--out_root", str(out_root),
            "--train_threshold", str(args.train_threshold),
            "--max_length", str(args.max_length),
        ]
        run(cmd, cwd=repo_root)

    print("\nDone.")
    print("Next steps:")
    print(f"  - Raw data:        {casp_dir}")
    print(f"  - Processed data:  {out_root / args.casp}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
