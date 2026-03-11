# Hybrid Protein Backbone Prediction (ProteinNet Transformer)

A lightweight Transformer trained on **ProteinNet CASP12** to predict **Cα backbone coordinates** from protein sequence.

This repo is packaged so that **anyone can run inference on their own FASTA files** on a laptop using Docker (recommended), or without Docker if they prefer.

**Model weights**
- Included locally: `artifacts/best.pt` + `artifacts/config_resolved.yaml`
- Hosted on Hugging Face Hub: `Milan890/proteinnet-hybrid-ca` (`best.pt`)

> Output is **Cα-only PDB** (coarse backbone). This is intended for fast visualization/triage and as an ML systems project—not AlphaFold-level atomic accuracy.

---

## Table of Contents
1. [Requirements](#1-requirements)  
2. [Build the Docker image](#2-build-the-docker-image)  
3. [Run inference on YOUR files (Docker)](#3-run-inference-on-your-files-docker)  
   3.1 [Modes: Lite vs MSA](#31-modes-lite-vs-msa-how-to-switch)  
   3.2 [Linux/macOS: Lite mode](#32-linuxmacos-lite-mode-fasta-only)  
   3.3 [Linux/macOS: MSA mode](#33-linuxmacos-msa-mode-a3m-provided)  
   3.4 [Linux/macOS: Offline mode](#34-linuxmacos-offline-mode-no-hf-download)  
   3.5 [Windows PowerShell: Lite mode](#35-windows-powershell-lite-mode)  
   3.6 [Optional: HF token](#36-optional-hugging-face-token-rate-limits)  
4. [Verify outputs](#4-verify-outputs)  
5. [Run without Docker (optional)](#5-run-without-docker-optional)  
6. [Visualization (PyMOL)](#6-visualization-pymol)  
7. [Troubleshooting](#7-troubleshooting)  
8. [Notes & limitations](#8-notes--limitations)

---

## 1) Requirements

### macOS / Linux
- Docker Desktop installed and running (recommended)
- Python 3.9+ only needed if running without Docker

### Windows
- Docker Desktop installed and running
- PowerShell

---

## 2) Build the Docker image

From the repo root (folder containing `docker/`, `src/`, `predict.py`, `artifacts/`):

```bash
docker build -f docker/Dockerfile.cpu -t proteinnet-hybrid:cpu .
````

If you see `docker: command not found`, install Docker Desktop and restart your terminal.

---

## 3) Run inference on YOUR files (Docker)

### How Docker paths work (important)

A container can’t see host files unless you mount folders using:

* `-v <HOST_FOLDER>:<CONTAINER_FOLDER>`

After mounting, you reference files using the container paths.

Example:

* Host folder: `/Users/alex/data/proteins`
* Mounted into container as `/in`
* FASTA path inside container: `/in/my_sequences.fasta`

---

## 3.1) Modes: Lite vs MSA (how to switch)

### Lite mode (FASTA-only)

Use when you only have a FASTA file.

* Use: `--mode lite`
* Do **NOT** pass `--a3m`

### MSA mode (A3M provided)

Use when you also have an A3M alignment file for the same sequence(s).

* Use: `--mode msa`
* Must include: `--a3m /in/<YOUR_A3M_FILENAME>.a3m`

✅ Switching modes is just:

* change `--mode lite` → `--mode msa`
* add `--a3m ...`

---

## 3.2) Linux/macOS: Lite mode (FASTA-only)

Copy-paste and edit only the 3 variables at the top:

```bash
# ---- EDIT THESE 3 LINES ----
INPUT_DIR="/Users/<you>/data/proteins"          # folder containing your FASTA
OUTPUT_DIR="/Users/<you>/results/proteinnet1"   # outputs go here
FASTA_NAME="my_sequences.fasta"                 # your FASTA filename
# ----------------------------

mkdir -p "$OUTPUT_DIR"

docker run --rm \
  -v "$INPUT_DIR:/in" \
  -v "$OUTPUT_DIR:/out" \
  proteinnet-hybrid:cpu \
  --hf_repo Milan890/proteinnet-hybrid-ca \
  --hf_filename best.pt \
  --config /app/artifacts/config_resolved.yaml \
  --fasta "/in/$FASTA_NAME" \
  --out_dir /out \
  --mode lite --device cpu
```

✅ Outputs are written into the host folder: `$OUTPUT_DIR`.

---

## 3.3) Linux/macOS: MSA mode (A3M provided)

Copy-paste and edit the 4 variables at the top:

```bash
# ---- EDIT THESE 4 LINES ----
INPUT_DIR="/Users/<you>/data/proteins"          # folder containing FASTA + A3M
OUTPUT_DIR="/Users/<you>/results/proteinnet_msa"
FASTA_NAME="my_sequences.fasta"
A3M_NAME="my_sequences.a3m"
# ----------------------------

mkdir -p "$OUTPUT_DIR"

docker run --rm \
  -v "$INPUT_DIR:/in" \
  -v "$OUTPUT_DIR:/out" \
  proteinnet-hybrid:cpu \
  --hf_repo Milan890/proteinnet-hybrid-ca \
  --hf_filename best.pt \
  --config /app/artifacts/config_resolved.yaml \
  --fasta "/in/$FASTA_NAME" \
  --a3m "/in/$A3M_NAME" \
  --out_dir /out \
  --mode msa --device cpu
```

---

## 3.4) Linux/macOS: Offline mode (no HF download)

Runs without Hugging Face by mounting this repo’s `artifacts/` folder:

```bash
# ---- EDIT THESE 3 LINES ----
INPUT_DIR="/Users/<you>/data/proteins"
OUTPUT_DIR="/Users/<you>/results/proteinnet_offline"
FASTA_NAME="my_sequences.fasta"
# ----------------------------

mkdir -p "$OUTPUT_DIR"

docker run --rm \
  -v "$INPUT_DIR:/in" \
  -v "$OUTPUT_DIR:/out" \
  -v "$PWD/artifacts:/artifacts" \
  proteinnet-hybrid:cpu \
  --checkpoint /artifacts/best.pt \
  --config /artifacts/config_resolved.yaml \
  --fasta "/in/$FASTA_NAME" \
  --out_dir /out \
  --mode lite --device cpu
```

---

## 3.5) Windows PowerShell: Lite mode

Edit the 3 variables and run:

```powershell
# ---- EDIT THESE 3 LINES ----
$INPUT_DIR  = "C:\Users\<you>\data\proteins"
$OUTPUT_DIR = "C:\Users\<you>\results\proteinnet1"
$FASTA_NAME = "my_sequences.fasta"
# ----------------------------

New-Item -ItemType Directory -Force -Path $OUTPUT_DIR | Out-Null

docker run --rm `
  -v "${INPUT_DIR}:/in" `
  -v "${OUTPUT_DIR}:/out" `
  proteinnet-hybrid:cpu `
  --hf_repo Milan890/proteinnet-hybrid-ca `
  --hf_filename best.pt `
  --config /app/artifacts/config_resolved.yaml `
  --fasta "/in/$FASTA_NAME" `
  --out_dir /out `
  --mode lite --device cpu
```

---

## 3.6) Optional: Hugging Face token (rate limits)

If you see Hugging Face rate-limit warnings, add a token:

```bash
docker run --rm \
  -e HF_TOKEN=hf_xxx \
  -v "$INPUT_DIR:/in" \
  -v "$OUTPUT_DIR:/out" \
  proteinnet-hybrid:cpu \
  --hf_repo Milan890/proteinnet-hybrid-ca \
  --hf_filename best.pt \
  --config /app/artifacts/config_resolved.yaml \
  --fasta "/in/$FASTA_NAME" \
  --out_dir /out \
  --mode lite --device cpu
```

---

## 4) Verify outputs

After a run, your output folder contains:

* `<record_id>_pred_ca.pdb`
* `<record_id>_meta.json`
* `manifest.json`

### Linux/macOS

```bash
ls -lah "$OUTPUT_DIR"
head -n 5 "$OUTPUT_DIR"/*_pred_ca.pdb
```

### Windows PowerShell

```powershell
Get-ChildItem $OUTPUT_DIR
Get-Content (Join-Path $OUTPUT_DIR "*_pred_ca.pdb") -TotalCount 5
```

---

## 5) Run without Docker (optional)

Install dependencies:

```bash
pip install torch numpy pyyaml tqdm huggingface_hub
```

Lite mode (local checkpoint):

```bash
python predict.py \
  --checkpoint artifacts/best.pt \
  --config artifacts/config_resolved.yaml \
  --fasta /path/to/your.fasta \
  --out_dir /path/to/output_folder \
  --mode lite --device cpu
```

Lite mode (download from HF):

```bash
python predict.py \
  --hf_repo Milan890/proteinnet-hybrid-ca \
  --hf_filename best.pt \
  --config artifacts/config_resolved.yaml \
  --fasta /path/to/your.fasta \
  --out_dir /path/to/output_folder \
  --mode lite --device cpu
```

---

## 6) Visualization (PyMOL)

Outputs are Cα-only PDB files. Open them in PyMOL:

1. File → Open → `<record_id>_pred_ca.pdb`
2. (Optional) Load a reference structure and overlay/align.

---

## 7) Troubleshooting

* `docker: command not found`
  → Install Docker Desktop and restart terminal.

* `Cannot connect to the Docker daemon`
  → Docker Desktop is not running yet.

* FASTA “file not found”
  → Verify mounts (`-v ...`) and filenames.

* Output folder empty
  → Ensure output mount and `--out_dir /out`.

---

## 8) Notes & limitations

* Predicts **Cα-only** coordinates (coarse backbone)
* Not intended for docking / atomic-level design
* MSA mode requires a user-provided A3M alignment; alignment generation is not bundled

```
