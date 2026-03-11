#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from src.data.proteinnet_parser import iter_records


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preprocess ProteinNet split files into memory-mapped .npy arrays.")
    p.add_argument("--casp", type=str, default="casp12", help="ProteinNet dataset name (casp7..casp12).")
    p.add_argument("--raw_root", type=Path, required=True, help="Path containing the extracted ProteinNet folder (e.g. data/raw).")
    p.add_argument("--out_root", type=Path, required=True, help="Output root for processed arrays (e.g. data/processed).")
    p.add_argument("--train_threshold", type=int, default=90, help="Training thinning threshold (30/50/70/90/95/100).")
    p.add_argument("--max_length", type=int, default=1024, help="Truncate sequences longer than this. Use 0 to disable.")
    p.add_argument("--max_records", type=int, default=0, help="For debugging: only process first N records per split (0 = all).")
    return p.parse_args()


def resolve_casp_dir(raw_root: Path, casp: str) -> Path:
    # Most extractions create raw_root/casp12/...
    if (raw_root / casp).exists():
        return raw_root / casp
    # Sometimes the user points directly at the casp folder.
    return raw_root


def compute_index(
    split_path: Path,
    *,
    max_length: Optional[int],
    max_records: Optional[int],
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """First pass: compute per-record lengths and offsets."""
    ids: List[str] = []
    lengths: List[int] = []

    n_seen = 0
    for rec in tqdm(iter_records(split_path), desc=f"Indexing {split_path.name}", unit="rec"):
        L = rec.length
        if max_length is not None and L > max_length:
            L = max_length
        ids.append(rec.id)
        lengths.append(int(L))

        n_seen += 1
        if max_records and n_seen >= max_records:
            break

    lengths_arr = np.asarray(lengths, dtype=np.int32)
    offsets = np.zeros((len(lengths_arr) + 1,), dtype=np.int64)
    offsets[1:] = np.cumsum(lengths_arr, dtype=np.int64)
    return ids, lengths_arr, offsets


def write_split(
    split_path: Path,
    out_dir: Path,
    *,
    max_length: Optional[int],
    max_records: Optional[int],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    ids, lengths, offsets = compute_index(split_path, max_length=max_length, max_records=max_records)
    total_len = int(offsets[-1])

    print(f"[{split_path.name}] records={len(lengths)} total_residues={total_len}")

    # Allocate memory-mapped .npy arrays
    tokens_mm = np.lib.format.open_memmap(out_dir / "tokens.npy", mode="w+", dtype=np.int16, shape=(total_len,))
    evo_mm = np.lib.format.open_memmap(out_dir / "evo.npy", mode="w+", dtype=np.float16, shape=(total_len, 21))
    ca_mm = np.lib.format.open_memmap(out_dir / "ca_coords.npy", mode="w+", dtype=np.float32, shape=(total_len, 3))
    mask_mm = np.lib.format.open_memmap(out_dir / "mask.npy", mode="w+", dtype=np.uint8, shape=(total_len,))

    # Second pass: fill arrays
    cursor = 0
    n_seen = 0
    for rec in tqdm(iter_records(split_path), desc=f"Writing {split_path.name}", unit="rec"):
        L = rec.length
        if max_length is not None and L > max_length:
            L = max_length

        tokens_mm[cursor : cursor + L] = rec.tokens[:L].astype(np.int16, copy=False)
        evo_mm[cursor : cursor + L, :] = rec.evolutionary[:L, :].astype(np.float16, copy=False)
        ca_mm[cursor : cursor + L, :] = rec.ca_coords[:L, :].astype(np.float32, copy=False)
        mask_mm[cursor : cursor + L] = rec.mask[:L].astype(np.uint8, copy=False)

        cursor += L
        n_seen += 1
        if max_records and n_seen >= max_records:
            break

    assert cursor == total_len, f"Cursor mismatch: cursor={cursor} total_len={total_len}"

    np.save(out_dir / "lengths.npy", lengths)
    np.save(out_dir / "offsets.npy", offsets)
    (out_dir / "ids.txt").write_text("\n".join(ids) + "\n")

    print(f"Wrote {out_dir}")


def main() -> None:
    args = parse_args()
    casp_dir = resolve_casp_dir(args.raw_root, args.casp)

    max_length = None if args.max_length in (0, None) else int(args.max_length)
    max_records = None if args.max_records in (0, None) else int(args.max_records)

    train_file = casp_dir / f"training_{args.train_threshold}"
    val_file = casp_dir / "validation"
    test_file = casp_dir / "testing"

    if not train_file.exists():
        raise FileNotFoundError(f"Missing train split: {train_file}")
    if not val_file.exists():
        raise FileNotFoundError(f"Missing validation split: {val_file}")
    if not test_file.exists():
        raise FileNotFoundError(f"Missing testing split: {test_file}")

    out_root = args.out_root / args.casp
    write_split(train_file, out_root / f"train_{args.train_threshold}", max_length=max_length, max_records=max_records)
    write_split(val_file, out_root / "validation", max_length=max_length, max_records=max_records)
    write_split(test_file, out_root / "testing", max_length=max_length, max_records=max_records)


if __name__ == "__main__":
    main()
