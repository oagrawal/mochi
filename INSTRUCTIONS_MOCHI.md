# Mochi + VBench Evaluation — Run Instructions

This document describes how to run the **same experiment** as Wan2.1 and HunyuanVideo: generate videos for the 33 VBench prompts and run the full VBench + fidelity pipeline. Mochi has **one mode** (`mochi_baseline`) in this setup; if TeaCache adds Mochi support later, you can add fixed/adaptive modes.

**Important:** Mochi uses a **different environment** (uv, different deps) than HunyuanVideo and Wan2.1. Use a **separate Docker container or venv** for Mochi. VBench evaluation reuses the **HunyuanVideo eval container** (same as for Wan2.1).

**Per-machine vs shared:** Docker **containers** and **images** exist on each host. The **repo, weights, and outputs** can live on a shared mount (e.g. `/nfs/oagrawal/mochi`). So: on a **new machine** you must **create the container again** and **install dependencies inside it**; you do **not** need to re-clone or re-download weights if the same path is mounted.

---

## File system layout (target state)

After cloning and copying, you should have:

```
/nfs/oagrawal/mochi/                    # Mochi repo root (clone here)
├── demos/
├── scripts/
├── genmo/
├── weights/                            # Model weights (download here)
│   ├── dit.safetensors
│   └── decoder.safetensors
├── vbench_eval/                        # Copied from mochi_experiment
│   ├── prompts_subset.json             # 33 VBench prompts (same as Wan/Hunyuan)
│   ├── batch_generate_mochi.py         # Batch video generation
│   ├── videos/
│   │   └── mochi_baseline/             #   {prompt}-{seed}.mp4
│   ├── vbench_scores/                  # Filled by Step 2 (HunyuanVideo container)
│   │   └── mochi_baseline/
│   ├── fidelity_metrics/              # Optional for single mode
│   ├── all_comparison_results.json
│   ├── vbench_scores_table.csv
│   ├── fidelity_table.csv
│   └── summary_table.csv
├── teacache_mochi.py                    # TeaCache integration for Mochi (diffusers)
├── INSTRUCTIONS_MOCHI.md               # This file (copy from mochi_experiment)
└── README_MOCHI_EXPERIMENT.md          # Short overview (copy from mochi_experiment)
```

**Staging folder (already created):** `/nfs/oagrawal/mochi_experiment/` contains `vbench_eval/`, `INSTRUCTIONS_MOCHI.md`, `README_MOCHI_EXPERIMENT.md`, and `PLAN.md` (one-page summary). Copy the first four into the Mochi repo after cloning; PLAN.md is optional. The file `vbench_eval/prompts_subset.json` is kept in sync with HunyuanVideo by copying: `cp /nfs/oagrawal/HunyuanVideo/vbench_eval/prompts_subset.json /nfs/oagrawal/mochi_experiment/vbench_eval/prompts_subset.json`

---

## Step 0: Clone Mochi and set up structure

From the host:

```bash
# 1) Clone the official Mochi repo
cd /nfs/oagrawal
git clone https://github.com/genmoai/mochi

# 2) Copy experiment files from staging into the repo
cp -r /nfs/oagrawal/mochi_experiment/vbench_eval /nfs/oagrawal/mochi/
cp /nfs/oagrawal/mochi_experiment/INSTRUCTIONS_MOCHI.md /nfs/oagrawal/mochi/
cp /nfs/oagrawal/mochi_experiment/README_MOCHI_EXPERIMENT.md /nfs/oagrawal/mochi/

# Optional: use HunyuanVideo as single source of truth for prompts (instead of staging copy)
# cp /nfs/oagrawal/HunyuanVideo/vbench_eval/prompts_subset.json /nfs/oagrawal/mochi/vbench_eval/prompts_subset.json

# 3) Create weights directory (you will download into it)
mkdir -p /nfs/oagrawal/mochi/weights
```

---

## Step 1: Docker container for Mochi (recommended)

Mochi needs ~60GB VRAM for single-GPU (or multi-GPU). Use a PyTorch CUDA image and install Mochi inside. **The container is per host:** on each machine where you want to run Mochi, create the container once (and install deps inside it); see **Running on another machine** below if the repo is on a shared mount.

### Create and enter the container

From the **host** (on the machine where you will run Mochi):

```bash
docker run -it --gpus all --name mochi \
  -v /nfs/oagrawal/mochi:/workspace/mochi \
  pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel bash
```

Adjust `-v` if your Mochi path differs. Inside the container, the repo is at `/workspace/mochi`.

### Install Mochi inside the container

```bash
cd /workspace/mochi

# Option A: Use uv (as in official README)
pip install uv
uv venv .venv
source .venv/bin/activate
uv pip install setuptools
uv pip install -e . --no-build-isolation
# Optional: flash attention
# uv pip install -e .[flash] --no-build-isolation

# Install FFMPEG (required to save videos)
apt-get update && apt-get install -y ffmpeg

# For batch script (progress bar) — use same tool as above so it goes into .venv
uv pip install tqdm
```

**Option B: Plain pip (if uv causes issues)**

```bash
cd /workspace/mochi
pip install --upgrade pip
pip install setuptools
pip install -e . --no-build-isolation
apt-get update && apt-get install -y ffmpeg
pip install tqdm
```

**Dependencies in one place (Mochi container):** If you want to install everything with one tool and avoid mixing `pip` and `uv`, use either block below. This covers Mochi (Step 1), tqdm, FFMPEG, and TeaCache/diffusers/matplotlib (Step 2.5).

- **All with uv (recommended if you use Option A):**
  ```bash
  cd /workspace/mochi
  pip install uv && uv venv .venv && source .venv/bin/activate
  uv pip install setuptools && uv pip install -e . --no-build-isolation
  uv pip install tqdm "diffusers[torch]" transformers protobuf tokenizers sentencepiece imageio matplotlib
  apt-get update && apt-get install -y ffmpeg
  ```
- **All with pip (if you use Option B):**
  ```bash
  cd /workspace/mochi
  pip install --upgrade pip setuptools && pip install -e . --no-build-isolation
  pip install tqdm "diffusers[torch]" transformers protobuf tokenizers sentencepiece imageio matplotlib
  apt-get update && apt-get install -y ffmpeg
  ```

### Enter the container later

```bash
docker start -ai mochi
```

Extra shell:

```bash
docker exec -it mochi bash
```

### Running on another machine (same mounted filesystem)

If you move to a **different host** that mounts the same path (e.g. `/nfs/oagrawal/mochi`):

1. **Containers do not follow you.** On the new machine there is no `mochi` container yet. Create it with the **same** `docker run` as in Step 1 (same image, same `-v` so the mount is visible inside the container).

2. **Repo and weights:** If the filesystem is shared, the repo at `/nfs/oagrawal/mochi` (and `weights/`, `.venv/` if you created them there) are already visible on the new host. You do **not** re-clone or re-copy from `mochi_experiment`. You **do** need to install dependencies inside the new container (the `.venv` on the mount will be used if it exists and has the right packages; if the new host uses a different arch or you prefer a clean env, recreate the venv inside the container as in Step 1).

3. **One-time setup on the new host:**
   - Pull the image (if needed): `docker pull pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel`
   - Create and enter the container:  
     `docker run -it --gpus all --name mochi -v /nfs/oagrawal/mochi:/workspace/mochi pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel bash`
   - Inside the container: `cd /workspace/mochi && source .venv/bin/activate` (if `.venv` already exists and has deps) or run the full install block from Step 1 (uv/pip + FFMPEG + TeaCache deps).

4. **Then** run your commands (e.g. delta TEMNI, batch generation) as on the first machine. Use the same paths; outputs will go to the shared mount.

If the **path** to the repo is different on the new host (e.g. `/other/mount/mochi`), change the `-v` to that path so the container sees the repo and weights.

---

## Step 2: Download weights

Inside the Mochi container (or on host if repo is local):

```bash
cd /workspace/mochi
python3 ./scripts/download_weights.py weights/
```

**If you see "Not enough free disk space" for `/tmp`:** The host root disk may be full (~31G free); the script’s cache can use `/tmp` inside the container. Point cache and temp to the mounted repo (on NFS, which has plenty of space) before running:

```bash
cd /workspace/mochi
export HF_HOME=/workspace/mochi/.cache/huggingface
export TMPDIR=/workspace/mochi/.cache/tmp
mkdir -p "$TMPDIR" "$HF_HOME"
python3 ./scripts/download_weights.py weights/
```

Or download manually from [Hugging Face](https://huggingface.co/genmoai/mochi) or via the magnet link in the Mochi README into `weights/`.

Verify:

```bash
ls weights/
# Should show dit.safetensors, decoder.safetensors (and possibly other files)
```

---

## Step 2.5: Set up TeaCache (Mochi implementation)

The file `teacache_mochi.py` in the repo root adds TeaCache caching to Mochi via the **diffusers** pipeline. Install its dependencies and optionally run a single-video test before batch generation.

**Important:** Use the **same** installer as in Step 1 so packages go into the same venv. If you used **uv** (Option A), use `uv pip install` below; if you used **pip** (Option B), use `pip install`.

### Install TeaCache dependencies

Inside the Mochi container, with your venv activated:

```bash
cd /workspace/mochi
source .venv/bin/activate   # if using uv venv

# If you used Option A (uv) in Step 1 — use uv so deps go into the same .venv:
uv pip install "diffusers[torch]" transformers protobuf tokenizers sentencepiece imageio matplotlib

# If you used Option B (pip) in Step 1 — use pip:
# pip install --upgrade "diffusers[torch]" transformers protobuf tokenizers sentencepiece imageio matplotlib
```

### Configure the threshold

In `teacache_mochi.py`, the TeaCache threshold is set via **`--teacache_thresh`** (default `0.09`) or in code (`rel_l1_thresh`). It trades off latency vs visual quality:

- **`0.06`** — ~1.5× speedup, stricter (better quality, less skip)
- **`0.09`** — ~2.1× speedup (default in the script)

Edit the file and change the value if you want a different trade-off.

### Single-GPU inference with TeaCache

From the repo root:

```bash
cd /workspace/mochi
python teacache_mochi.py
```

The script writes an MP4 to the current directory (e.g. `teacache_mochi__A hand with delicate fingers....mp4`). It uses **diffusers** and loads the model via `MochiPipeline.from_pretrained("genmo/mochi-1-preview")`, so the first run may download from Hugging Face; subsequent runs use the cache.

### Delta TEMNI plot (no caching, baseline)

To get a **delta TEMNI over diffusion steps** plot (same idea as Wan2.1’s no-cache baseline), run with `--delta_temni_plot`. This disables TeaCache, runs every forward step, and records the rescaled relative L1 (delta TEMNI) at each step, then saves a plot and a `.txt` of values next to the video. (Matplotlib is installed in Step 2.5 above.)

Run (from inside the Mochi container):

```bash
cd /workspace/mochi
python3 teacache_mochi.py --delta_temni_plot --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." --out_dir ./mochi_results --save_file ./mochi_results/cats_boxing_baseline.mp4 --seed 42
```

If another process (e.g. VLLM) is using GPU 0, use the other GPU so the script has enough VRAM:

```bash
CUDA_VISIBLE_DEVICES=1 python3 teacache_mochi.py --delta_temni_plot --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." --out_dir ./mochi_results --save_file ./mochi_results/cats_boxing_baseline.mp4 --seed 42
```

**Tmux (SSH-safe):** The run takes ~30+ minutes. Start tmux on the **host** first (the container image may not have tmux), then attach to the container inside that session so you can reattach after an SSH drop:

```bash
# 1) On the host: start a new tmux session
tmux new -s mochi_delta

# 2) Inside tmux: enter the Mochi container
docker start -ai mochi

# 3) Inside the container: ensure venv and cwd, then run the command
cd /workspace/mochi && source .venv/bin/activate
CUDA_VISIBLE_DEVICES=1 python3 teacache_mochi.py --delta_temni_plot --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." --out_dir ./mochi_results --save_file ./mochi_results/cats_boxing_baseline.mp4 --seed 42

# 4) If SSH disconnects: reconnect on the host, then reattach to tmux
#    tmux attach -t mochi_delta
```

**Outputs** (in `mochi_results/`):

- `cats_boxing_baseline.mp4` — generated video
- `cats_boxing_baseline_delta_TEMNI_plot.png` — plot of delta TEMNI vs forward step (same style as Wan2.1)
- `cats_boxing_baseline_delta_TEMNI.txt` — one value per line

You can use any `--prompt` and `--save_file`; the plot and `.txt` use the same base name as the video.

---

## Step 3: Sanity check (single video)

From repo root inside the container:

```bash
cd /workspace/mochi
source .venv/bin/activate   # if using uv venv

python3 ./demos/cli.py --model_dir weights/ --cpu_offload \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  --out_dir ./test_out --seed 42
```

Check that `test_out/` contains an MP4. Then proceed to batch generation.

---

## Step 4: Batch generate videos (VBench 33 prompts)

**Mode:** Only `mochi_baseline` (no TeaCache for Mochi in this setup).

Use **tmux** for long runs (generation can take many hours).

```bash
# Terminal 1
tmux new -s mochi_gen0
docker start -ai mochi
cd /workspace/mochi && source .venv/bin/activate

# Dry run first
python3 vbench_eval/batch_generate_mochi.py --model_dir weights/ --cpu_offload --dry-run
```

**Single GPU (all 33 prompts):**

```bash
cd /workspace/mochi
CUDA_VISIBLE_DEVICES=0 python3 vbench_eval/batch_generate_mochi.py \
  --model_dir weights/ --cpu_offload \
  --output-dir vbench_eval/videos
```

**Split across 2 GPUs:**

**GPU 0** (prompts 0–16):

```bash
cd /workspace/mochi
CUDA_VISIBLE_DEVICES=0 python3 vbench_eval/batch_generate_mochi.py \
  --model_dir weights/ --cpu_offload \
  --output-dir vbench_eval/videos \
  --start-idx 0 --end-idx 17
```

**GPU 1** (prompts 17–32):

```bash
cd /workspace/mochi
CUDA_VISIBLE_DEVICES=1 python3 vbench_eval/batch_generate_mochi.py \
  --model_dir weights/ --cpu_offload \
  --output-dir vbench_eval/videos \
  --start-idx 17 --end-idx 33
```

**Resume:** Re-run the same command after Ctrl+C or restart; the script skips videos that already exist and appends to `generation_log_*.json`.

---

## Step 5: Run VBench evaluation (HunyuanVideo container)

VBench and fidelity scripts live in HunyuanVideo; they can read any path visible inside the container. Use the **eval container** that mounts `/nfs/oagrawal` (same as for Wan2.1).

### Ensure eval container can see Mochi

From host (one-time, if you don’t have it yet):

```bash
docker run -it --gpus all --init --net=host --uts=host --ipc=host \
  --name hunyuanvideo_eval_wan --security-opt=seccomp=unconfined \
  --ulimit=stack=67108864 --ulimit=memlock=-1 --privileged \
  -v /nfs/oagrawal:/nfs/oagrawal \
  hunyuanvideo/hunyuanvideo:cuda_11 bash
```

Re-enter: `docker start -ai hunyuanvideo_eval_wan`

Inside the eval container:

```bash
# Switch transformers for VBench
pip install transformers==4.33.2

# Ensure VBench and deps (from HunyuanVideo repo)
cd /nfs/oagrawal/HunyuanVideo
pip install -e ./VBench
pip install lpips
apt-get update && apt-get install -y libgl1 libglib2.0-0
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

Set paths (adjust if needed):

```bash
MOCHI_VBENCH=/nfs/oagrawal/mochi/vbench_eval
HV_ROOT=/nfs/oagrawal/HunyuanVideo
```

**Run VBench for Mochi (single mode):**

```bash
cd $HV_ROOT
CUDA_VISIBLE_DEVICES=0 python3 vbench_eval/run_vbench_eval.py \
  --video-dir $MOCHI_VBENCH/videos \
  --save-dir $MOCHI_VBENCH/vbench_scores \
  --full-info $MOCHI_VBENCH/prompts_subset.json \
  --modes mochi_baseline
```

---

## Step 6: Fidelity metrics (optional for single mode)

With only `mochi_baseline`, there is no “cached” mode to compare. You can skip this step, or run it for future use when you add more modes. To run (baseline vs itself is trivial):

```bash
cd $HV_ROOT
CUDA_VISIBLE_DEVICES=0 python3 vbench_eval/run_fidelity_metrics.py \
  --video-dir $MOCHI_VBENCH/videos \
  --baseline mochi_baseline \
  --modes mochi_baseline \
  --save-dir $MOCHI_VBENCH/fidelity_metrics
```

---

## Step 7: Compare results (3 CSVs)

No GPU; run from host or eval container:

```bash
cd $HV_ROOT
python3 vbench_eval/compare_results.py \
  --scores-dir $MOCHI_VBENCH/vbench_scores \
  --fidelity-dir $MOCHI_VBENCH/fidelity_metrics \
  --gen-log-dir $MOCHI_VBENCH/videos \
  --output-json $MOCHI_VBENCH/all_comparison_results.json \
  --modes mochi_baseline
```

**Output files** (under `$MOCHI_VBENCH/`):

| File | Description |
|------|-------------|
| `vbench_scores_table.csv` | VBench 16 dimensions + latency |
| `fidelity_table.csv` | PSNR/SSIM/LPIPS (single mode: one row) |
| `summary_table.csv` | Compact summary |

---

## Quick reference

| Step | Where | Command / note |
|------|--------|-----------------|
| 0 | Host | Clone Mochi to `/nfs/oagrawal/mochi`, copy `mochi_experiment` contents into it |
| 1 | Host | Create Docker container `mochi` with mount to `/workspace/mochi` |
| 2 | Mochi container | Install Mochi (uv or pip), FFMPEG; download weights to `weights/` |
| 2.5 | Mochi container | TeaCache: `pip install diffusers[torch] transformers ... matplotlib`; run `python teacache_mochi.py` or `python teacache_mochi.py --delta_temni_plot ...` for delta TEMNI plot |
| 3 | Mochi container | Sanity check: `demos/cli.py --model_dir weights/ --cpu_offload ...` |
| 4 | Mochi container, tmux | `batch_generate_mochi.py` with `--model_dir weights/`, optional `--start-idx` / `--end-idx` |
| 5 | HunyuanVideo eval container | `run_vbench_eval.py` with `--video-dir` / `--save-dir` / `--full-info` = `$MOCHI_VBENCH`, `--modes mochi_baseline` |
| 6 | HunyuanVideo eval container | Optional: `run_fidelity_metrics.py` |
| 7 | Any | `compare_results.py` with `--modes mochi_baseline` and paths to `$MOCHI_VBENCH` |

---

## Reference prompt (same as Wan/Hunyuan)

```
Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.
```

---

## Hardware note

Mochi requires ~60GB VRAM on a single GPU (recommended: H100). Use `--cpu_offload` to reduce VRAM; multi-GPU is supported by the repo (see Mochi README).
