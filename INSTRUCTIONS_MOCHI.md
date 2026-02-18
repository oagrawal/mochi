# Mochi + VBench Evaluation — Run Instructions

This document describes how to run the **same experiment** as Wan2.1 and HunyuanVideo: generate videos for the 33 VBench prompts and run the full VBench + fidelity pipeline. Mochi has **one mode** (`mochi_baseline`) in this batch setup. **Note:** TeaCache IS implemented for Mochi (see `teacache_mochi.py` using the diffusers pipeline), but the batch script (`batch_generate_mochi.py`) currently only supports baseline mode because it uses the native Mochi pipeline (`MochiSingleGPUPipeline`). To add TeaCache modes to batch generation, you would need to modify the batch script to use the diffusers pipeline instead.

**Important:** Mochi uses a **different environment** (uv, different deps) than HunyuanVideo and Wan2.1. Use a **separate Docker container or venv** for Mochi. VBench evaluation reuses the **HunyuanVideo eval container** (same as for Wan2.1).

**Per-machine vs shared:** Docker **containers** and **images** exist on each host. The **repo, weights, and outputs** can live on a shared mount (e.g. `/nfs/oagrawal/mochi`). So: on a **new machine** you must **create the container again** and **install dependencies inside it**; you do **not** need to re-clone or re-download weights if the same path is mounted.

---

## Mini-experiment: TeaCache adaptive threshold selection

To choose **high** and **low** thresholds for a later multi-hour adaptive run, run this mini-experiment first. It generates **2 prompts × (1 baseline + 4 adaptive configs) = 10 videos** (same seed), then you run fidelity in the HunyuanVideo eval container to get **4 fidelity summaries** (PSNR/SSIM/LPIPS averaged over the 2 prompts). Pick the config with the best fidelity for the full run.

**Threshold interface in `teacache_mochi.py`:** Use `--teacache_thresh_low` and `--teacache_thresh_high`. If they are **equal** → fixed threshold for all steps. If **different** → adaptive: first 33 steps use high, rest use low. You can omit them and use `--teacache_thresh` for a single fixed threshold (backward compatible).

**Step 1 — Generate videos (Mochi container)**

Run inside the **Mochi Docker container** (see **Step 1: Docker container for Mochi** below) to avoid CUDA/driver issues. **On a new machine:** check if the container exists (`docker ps -a --filter name=mochi`); if not, create it with the `docker run` in Step 1, then install deps (uv + Mochi + TeaCache) inside. Use **tmux** (or four separate terminals) for session safety; in each terminal, attach to the **same** container, then use the **uv-created venv** and run one of the four commands.

There are **10 jobs** total (2 baseline + 8 adaptive). For **4 GPUs**, use the **runtime-balanced** split: **no GPU runs both baselines** (baseline is slowest), and each GPU that runs a baseline also runs an **aggressive** cache config (c2 or c4), which is faster, so wall time is more even.

| GPU | Jobs (by index) | Videos (runtime-balanced) |
|-----|------------------|---------------------------|
| 0   | 0, 4, 7         | baseline p0 + **c2** p0 + c3 p1 (1 base + 1 aggressive + 1 conservative) |
| 1   | 1, 8            | baseline p1 + **c4** p0 (1 base + 1 aggressive) |
| 2   | 5, 9, 2         | **c2** p1 + **c4** p1 + c1 p0 (2 aggressive + 1 conservative) |
| 3   | 3, 6            | c1 p1 + c3 p0 (2 conservative) |

Job order: 0=base p0, 1=base p1, 2=c1 p0, 3=c1 p1, 4=c2 p0, 5=c2 p1, 6=c3 p0, 7=c3 p1, 8=c4 p0, 9=c4 p1. Use **`--gpu-id`** and **`--total-gpus 4`** so the script assigns the above job sets automatically.

**Pre-download the model once (avoid 4× download):** Each process loads `MochiPipeline.from_pretrained("genmo/mochi-1-preview")`, so if you start all 4 GPU processes at once, each will download the same ~38GB model. **Do this once** in a single container shell before starting the four GPU runs:

```bash
docker exec -it mochi bash
cd /workspace/mochi && source .venv/bin/activate
python3 -c "from diffusers import MochiPipeline; MochiPipeline.from_pretrained('genmo/mochi-1-preview'); print('Model cached.')"
exit
```

After that, start the four GPU processes; they will use the cached model and not re-download.

**On the host:** start tmux (e.g. four panes or four windows), then in **each** terminal:

1. Start the container if it is not running: `docker start mochi`
2. Attach to the container: `docker exec -it mochi bash`
3. Inside the container: `cd /workspace/mochi`, activate the venv you created with **uv** (Step 1 below), then run the command for that GPU.

**Terminal 1 (GPU 0):**
```bash
# On host (optional, for session safety)
tmux new -s mochi_mini_0

# Attach to Mochi container (if not already inside)
docker start mochi
docker exec -it mochi bash

# Inside container: uv venv and run
cd /workspace/mochi
source .venv/bin/activate   # venv created with uv (Step 1)
CUDA_VISIBLE_DEVICES=0 python3 vbench_eval/run_adaptive_threshold_experiment.py --seed 0 --gpu-id 0 --total-gpus 4
```

**Terminal 2 (GPU 1):**
```bash
tmux new -s mochi_mini_1
docker exec -it mochi bash
cd /workspace/mochi
source .venv/bin/activate   # venv created with uv (Step 1)
CUDA_VISIBLE_DEVICES=1 python3 vbench_eval/run_adaptive_threshold_experiment.py --seed 0 --gpu-id 1 --total-gpus 4
```

**Terminal 3 (GPU 2):**
```bash
tmux new -s mochi_mini_2
docker exec -it mochi bash
cd /workspace/mochi
source .venv/bin/activate   # venv created with uv (Step 1)
CUDA_VISIBLE_DEVICES=2 python3 vbench_eval/run_adaptive_threshold_experiment.py --seed 0 --gpu-id 2 --total-gpus 4
```

**Terminal 4 (GPU 3):**
```bash
tmux new -s mochi_mini_3
docker exec -it mochi bash
cd /workspace/mochi
source .venv/bin/activate   # venv created with uv (Step 1)
CUDA_VISIBLE_DEVICES=3 python3 vbench_eval/run_adaptive_threshold_experiment.py --seed 0 --gpu-id 3 --total-gpus 4
```

You only need to run `docker start mochi` once (e.g. in terminal 1); the other terminals use `docker exec -it mochi bash` to get a new shell in the same running container.

**Next — run fidelity:** After all 10 videos are generated, run the fidelity step in the **HunyuanVideo eval container** (see **Step 2 — Fidelity metrics** below). On the host: `docker ps -a --filter name=hunyuanvideo_eval_wan` to check for the eval container; if missing, create it with the `docker run` in Step 5, then inside it run the `run_fidelity_metrics.py` command from Step 2.

**Single GPU:** omit `--gpu-id` / `--total-gpus` and run all 10 jobs on one GPU (inside the container, with venv activated):

```bash
cd /workspace/mochi
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python3 vbench_eval/run_adaptive_threshold_experiment.py --seed 0
```

**Manual job range:** you can still use `--start-idx` and `--end-idx` (e.g. `--start-idx 0 --end-idx 3`) if you prefer a custom split instead of the balanced 4-GPU assignment.

**Race conditions:** None. Each of the 10 jobs writes to a different file (`{mode_name}/{prompt}-{seed}.mp4`), and the 4-GPU split assigns each job to exactly one process. There is no shared log file or other shared writable state; only read-only use of `prompts_subset.json`. You can start all four processes at once.

Optional: add `--dry-run` to print which jobs would run. Outputs go to `vbench_eval/videos_threshold_experiment/` (subdirs: `mochi_baseline`, `mochi_adaptive_c1` … `mochi_adaptive_c4`). The script skips videos that already exist, so you can re-run or resume safely.

**Step 2 — Fidelity metrics (HunyuanVideo eval container)**

Use bash only for any copy/path setup (do not rely on LLM-generated file content). Inside the eval container:

```bash
MOCHI_VBENCH=/nfs/oagrawal/mochi/vbench_eval
HV_ROOT=/nfs/oagrawal/HunyuanVideo
cd $HV_ROOT
CUDA_VISIBLE_DEVICES=0 python3 vbench_eval/run_fidelity_metrics.py \
  --video-dir $MOCHI_VBENCH/videos_threshold_experiment \
  --baseline mochi_baseline \
  --modes mochi_adaptive_c1,mochi_adaptive_c2,mochi_adaptive_c3,mochi_adaptive_c4 \
  --save-dir $MOCHI_VBENCH/fidelity_metrics_threshold_experiment
```

**Step 3 — Use results**

Fidelity outputs (e.g. `fidelity_metrics_threshold_experiment/all_fidelity_results.json` and per-mode JSONs) give PSNR/SSIM/LPIPS **averaged over the 2 prompts** for each config. Choose the config with the best scores for your larger run.

---

## Full TeaCache 33-prompt batch experiment (4 modes × 33 prompts)

This section describes the **large batch job**: generate videos for the 33 VBench prompts with **4 modes** using the diffusers + TeaCache pipeline, then run VBench and fidelity. Modes:

- **`mochi_diff_baseline`** — diffusers baseline (no TeaCache)  
- **`mochi_fixed_0.04`** — fixed TeaCache threshold 0.04 (all steps)  
- **`mochi_fixed_0.12`** — fixed TeaCache threshold 0.12 (all steps)  
- **`mochi_adaptive_0.12_0.04`** — adaptive TeaCache: first 33 steps high=0.12, rest low=0.04  

All modes use the **same seed** (default `0`) and the same prompt set (`prompts_subset.json`) so fidelity and VBench are comparable across modes.

### Step 1 — Generate 33 × 4 videos (Mochi container, 4 GPUs)

Use the **Mochi Docker container** and the **uv** venv (same as the mini-experiment). Generation uses `vbench_eval/batch_generate_mochi_teacache.py`, which:

- Reads `vbench_eval/prompts_subset.json` (33 prompts)
- Generates videos into `vbench_eval/videos/{mode}/{prompt}-{seed}.mp4`
- Logs timings in `vbench_eval/videos/generation_log_teacache_{start}-{end}.json`
- Supports resume: existing `.mp4` files are skipped

**Prompts split across 4 GPUs:** 33 prompts → split as **8 + 8 + 8 + 9** so that 3 GPUs get 8 prompts and one gets 9 (more balanced). Each GPU runs **all 4 modes** for its prompt range; no overlapping `--start-idx` / `--end-idx`, so there are **no race conditions** and each video is generated once.

| GPU | Prompt indices (start–end) | Prompts count |
|-----|----------------------------|---------------|
| 0   | 0–8                        | 8             |
| 1   | 8–16                       | 8             |
| 2   | 16–24                      | 8             |
| 3   | 24–33                      | 9             |

**On the host:** as with the mini-experiment, use tmux (one session per GPU), then attach to the **same `mochi` container** in each terminal and run the command for that GPU.

- If you **already created** tmux sessions for the mini-experiment (e.g. `mochi_mini_0` … `mochi_mini_3`), you can **reuse them** instead of creating new ones:
  - List sessions: `tmux ls`
  - Reattach: `tmux attach -t mochi_mini_0` (or `mochi_mini_1`, `mochi_mini_2`, `mochi_mini_3`)
- If you prefer fresh sessions for this run, you can use names like `mochi_teacache_0` … `mochi_teacache_3` as below.

**Terminal 1 (GPU 0, prompts 0–7):**

```bash
# On host: either reattach an existing mini-experiment session...
#   tmux attach -t mochi_mini_0
# ...or create a new one:
tmux new -s mochi_teacache_0

# Attach to Mochi container (if not already inside)
docker start mochi
docker exec -it mochi bash

# Inside container: uv venv and run
cd /workspace/mochi
source .venv/bin/activate   # venv created with uv (Step 1)
CUDA_VISIBLE_DEVICES=0 python3 vbench_eval/batch_generate_mochi_teacache.py \
  --output-dir vbench_eval/videos \
  --generation-seed 0 \
  --start-idx 0 --end-idx 8
```

**Terminal 2 (GPU 1, prompts 8–15):**

```bash
# On host: reattach or create
#   tmux attach -t mochi_mini_1
tmux new -s mochi_teacache_1
docker exec -it mochi bash
cd /workspace/mochi
source .venv/bin/activate   # venv created with uv (Step 1)
CUDA_VISIBLE_DEVICES=1 python3 vbench_eval/batch_generate_mochi_teacache.py \
  --output-dir vbench_eval/videos \
  --generation-seed 0 \
  --start-idx 8 --end-idx 16
```

**Terminal 3 (GPU 2, prompts 16–23):**

```bash
# On host: reattach or create
#   tmux attach -t mochi_mini_2
tmux new -s mochi_teacache_2
docker exec -it mochi bash
cd /workspace/mochi
source .venv/bin/activate   # venv created with uv (Step 1)
CUDA_VISIBLE_DEVICES=2 python3 vbench_eval/batch_generate_mochi_teacache.py \
  --output-dir vbench_eval/videos \
  --generation-seed 0 \
  --start-idx 16 --end-idx 24
```

**Terminal 4 (GPU 3, prompts 24–32):**

```bash
# On host: reattach or create
#   tmux attach -t mochi_mini_3
tmux new -s mochi_teacache_3
docker exec -it mochi bash
cd /workspace/mochi
source .venv/bin/activate   # venv created with uv (Step 1)
CUDA_VISIBLE_DEVICES=3 python3 vbench_eval/batch_generate_mochi_teacache.py \
  --output-dir vbench_eval/videos \
  --generation-seed 0 \
  --start-idx 24 --end-idx 33
```

**Notes:**

- **Resume:** if a process is interrupted, re-run the same command; existing videos are skipped and the generation log is updated, so the run continues where it left off.
- **Timing data:** per-video timings are stored in `vbench_eval/videos/generation_log_teacache_{start}-{end}.json`, one log file per `start-idx`/`end-idx` range.
- **Seed:** keep `--generation-seed 0` (or your chosen seed) **fixed** across all GPUs and modes for clean fidelity comparisons.

**Operational details (learned from runs):**

- **Same container, other terminals:** Once one terminal has started the container (`docker start mochi`), other terminals use `docker exec -it mochi bash` to get a shell in the **same** container. Do **not** use `docker start -ai mochi` in every terminal; that attaches to the primary process. Use `docker exec -it mochi bash` for extra shells.
- **Pre-download model once:** If you start all 4 GPU processes at once, each would download the ~38GB model. Pre-download in one terminal:  
  `python3 -c "from diffusers import MochiPipeline; MochiPipeline.from_pretrained('genmo/mochi-1-preview'); print('Model cached.')"`  
  Then start the 4 processes; they will use the cached model.
- **GPU visibility:** Set `CUDA_VISIBLE_DEVICES=N` for each process. If you see `enable_model_cpu_offload requires accelerator, but not found`, the process is not seeing a GPU — ensure you are inside the container (which has GPU access) and that `CUDA_VISIBLE_DEVICES` matches an available GPU.
- **Re-splitting remaining work:** If one GPU (e.g. 8–16) fails partway, you can Ctrl+C and re-split the **remaining** prompts across GPUs using disjoint `--start-idx`/`--end-idx` ranges. Each range gets its own log file (`generation_log_teacache_{start}-{end}.json`), so no race conditions. Example: if prompts 13–15 remain from 8–16, run `--start-idx 13 --end-idx 14` on GPU 1, `--start-idx 14 --end-idx 15` on GPU 2, `--start-idx 15 --end-idx 16` on GPU 3.

**Single GPU:** to run all 33 × 4 videos on one GPU:

```bash
cd /workspace/mochi
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python3 vbench_eval/batch_generate_mochi_teacache.py \
  --output-dir vbench_eval/videos \
  --generation-seed 0
```

### Step 2 — VBench for the 4 modes (HunyuanVideo eval container)

Run VBench in the **HunyuanVideo eval container**. Each mode writes to its own subdirectory: `vbench_scores_teacache/{mode}/{dimension}_eval_results.json`, so splitting by **mode** across GPUs is safe — no shared writes, no race conditions.

**Scoring formula (used by `compare_results.py`):**  
`total_score = (quality_score × 4 + semantic_score × 1) / 5`  
(QUALITY_WEIGHT=4, SEMANTIC_WEIGHT=1)

**4-GPU split:** Assign one mode per GPU. Each process evaluates all 16 dimensions for that mode, writing only to its own `{save_dir}/{mode}/` folder.

**Resume support:** The VBench script skips dimensions that already have `{dimension}_eval_results.json` in the save path. If a run is interrupted or a terminal disconnects, re-run the same command in that tmux session; already-evaluated dimensions are skipped and work resumes from the next one.

| GPU | Mode |
|-----|------|
| 0   | `mochi_diff_baseline` |
| 1   | `mochi_fixed_0.04` |
| 2   | `mochi_fixed_0.12` |
| 3   | `mochi_adaptive_0.12_0.04` |

**Use tmux** so you can re-attach if a terminal disconnects or you need to detach:

```bash
# On host: create 4 tmux sessions (or reuse existing ones)
tmux new -s vbench_0
# ... run GPU 0 command below, then Ctrl+B D to detach ...

tmux new -s vbench_1
# ... run GPU 1 command below, then Ctrl+B D to detach ...

tmux new -s vbench_2
# ... run GPU 2 command below, then Ctrl+B D to detach ...

tmux new -s vbench_3
# ... run GPU 3 command below, then Ctrl+B D to detach ...
```

Inside the **HunyuanVideo eval container** in each tmux session:

**Terminal 1 (tmux vbench_0, GPU 0):**
```bash
docker start hunyuanvideo_eval_wan   # once; then use exec for extra shells
docker exec -it hunyuanvideo_eval_wan bash
MOCHI_VBENCH=/nfs/oagrawal/mochi/vbench_eval
HV_ROOT=/nfs/oagrawal/HunyuanVideo
cd $HV_ROOT
CUDA_VISIBLE_DEVICES=0 python3 vbench_eval/run_vbench_eval.py \
  --video-dir $MOCHI_VBENCH/videos \
  --save-dir $MOCHI_VBENCH/vbench_scores_teacache \
  --full-info $MOCHI_VBENCH/prompts_subset.json \
  --modes mochi_diff_baseline
```

**Terminal 2 (tmux vbench_1, GPU 1):**
```bash
docker exec -it hunyuanvideo_eval_wan bash
MOCHI_VBENCH=/nfs/oagrawal/mochi/vbench_eval
HV_ROOT=/nfs/oagrawal/HunyuanVideo
cd $HV_ROOT
CUDA_VISIBLE_DEVICES=1 python3 vbench_eval/run_vbench_eval.py \
  --video-dir $MOCHI_VBENCH/videos \
  --save-dir $MOCHI_VBENCH/vbench_scores_teacache \
  --full-info $MOCHI_VBENCH/prompts_subset.json \
  --modes mochi_fixed_0.04
```

**Terminal 3 (tmux vbench_2, GPU 2):**
```bash
docker exec -it hunyuanvideo_eval_wan bash
MOCHI_VBENCH=/nfs/oagrawal/mochi/vbench_eval
HV_ROOT=/nfs/oagrawal/HunyuanVideo
cd $HV_ROOT
CUDA_VISIBLE_DEVICES=2 python3 vbench_eval/run_vbench_eval.py \
  --video-dir $MOCHI_VBENCH/videos \
  --save-dir $MOCHI_VBENCH/vbench_scores_teacache \
  --full-info $MOCHI_VBENCH/prompts_subset.json \
  --modes mochi_fixed_0.12
```

**Terminal 4 (tmux vbench_3, GPU 3):**
```bash
docker exec -it hunyuanvideo_eval_wan bash
MOCHI_VBENCH=/nfs/oagrawal/mochi/vbench_eval
HV_ROOT=/nfs/oagrawal/HunyuanVideo
cd $HV_ROOT
CUDA_VISIBLE_DEVICES=3 python3 vbench_eval/run_vbench_eval.py \
  --video-dir $MOCHI_VBENCH/videos \
  --save-dir $MOCHI_VBENCH/vbench_scores_teacache \
  --full-info $MOCHI_VBENCH/prompts_subset.json \
  --modes mochi_adaptive_0.12_0.04
```

To reattach after disconnect: `tmux attach -t vbench_0` (or vbench_1, vbench_2, vbench_3).

**Single GPU, all 4 modes:** use one tmux session and:

```bash
CUDA_VISIBLE_DEVICES=0 python3 vbench_eval/run_vbench_eval.py \
  --video-dir $MOCHI_VBENCH/videos \
  --save-dir $MOCHI_VBENCH/vbench_scores_teacache \
  --full-info $MOCHI_VBENCH/prompts_subset.json \
  --modes mochi_diff_baseline,mochi_fixed_0.04,mochi_fixed_0.12,mochi_adaptive_0.12_0.04
```

This writes per-dimension VBench scores under `vbench_scores_teacache/{mode}/`.

### Step 3 — Fidelity metrics for the 4 modes (HunyuanVideo eval container)

Still inside the eval container, compute fidelity vs the diffusers baseline:

```bash
MOCHI_VBENCH=/nfs/oagrawal/mochi/vbench_eval
HV_ROOT=/nfs/oagrawal/HunyuanVideo
cd $HV_ROOT

CUDA_VISIBLE_DEVICES=0 python3 vbench_eval/run_fidelity_metrics.py \
  --video-dir $MOCHI_VBENCH/videos \
  --baseline mochi_diff_baseline \
  --modes mochi_fixed_0.04,mochi_fixed_0.12,mochi_adaptive_0.12_0.04 \
  --save-dir $MOCHI_VBENCH/fidelity_metrics_teacache
```

Outputs (under `fidelity_metrics_teacache/`):

- `mochi_fixed_0.04_vs_mochi_diff_baseline.json`
- `mochi_fixed_0.12_vs_mochi_diff_baseline.json`
- `mochi_adaptive_0.12_0.04_vs_mochi_diff_baseline.json`
- `all_fidelity_results.json` (summary over all modes)

### Step 4 — Compare results (optional summary CSV)

Use `compare_results.py` (from HunyuanVideo) to build comparison tables that include the new modes. From host or eval container:

```bash
MOCHI_VBENCH=/nfs/oagrawal/mochi/vbench_eval
HV_ROOT=/nfs/oagrawal/HunyuanVideo
cd $HV_ROOT

python3 vbench_eval/compare_results.py \
  --scores-dir $MOCHI_VBENCH/vbench_scores_teacache \
  --fidelity-dir $MOCHI_VBENCH/fidelity_metrics_teacache \
  --gen-log-dir $MOCHI_VBENCH/videos \
  --output-json $MOCHI_VBENCH/all_comparison_results_teacache.json \
  --modes mochi_diff_baseline,mochi_fixed_0.04,mochi_fixed_0.12,mochi_adaptive_0.12_0.04
```

This produces summary tables under `vbench_eval/` (e.g. CSVs with VBench + fidelity + speedup information) that you can use to choose the best TeaCache mode for the large run.

---

## What is implemented vs what needs to be implemented

**Implemented**

| Component | What it does |
|-----------|--------------|
| **`teacache_mochi.py`** | TeaCache for Mochi via the **diffusers** pipeline: single-video with `--teacache_thresh` (fixed) or `--teacache_thresh_low` / `--teacache_thresh_high` (fixed when equal, adaptive when different; first 33 steps high, rest low). Delta TEMNI plot with `--delta_temni_plot`. Uses `MochiPipeline.from_pretrained("genmo/mochi-1-preview")`. |
| **`batch_generate_mochi.py`** | Batch generation for the 33 VBench prompts using the **native** Mochi pipeline (`MochiSingleGPUPipeline` + local `weights/`). One mode only: `mochi_baseline` (no TeaCache). Supports `--start-idx` / `--end-idx`, resume, logging. |
| **VBench + fidelity pipeline** | Steps 5–7: run VBench evaluation and fidelity on `videos/`, compare results. Works for any mode(s) present under `videos/` (currently only `mochi_baseline`). |

**Not implemented (would need to be added)**

| Gap | What would be required |
|-----|-------------------------|
| **TeaCache in batch** | Batch script only uses the native pipeline. To run VBench with a TeaCache mode (e.g. `mochi_teacache`): either extend `batch_generate_mochi.py` with a mode/flag that uses the diffusers pipeline and the TeaCache logic from `teacache_mochi.py` (writing to e.g. `videos/mochi_teacache/`), or add a separate batch script that loops over prompts and calls `teacache_mochi.run_generation()` with the same seed and VBench naming `{prompt}-{seed}.mp4`. |
| **Fixed/adaptive TeaCache modes** | If you want multiple TeaCache variants (e.g. different thresholds or fixed vs adaptive), implement them in the chosen batch path above and add the corresponding subdirs and mode names to the VBench/fidelity steps. |

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

### Check if the Mochi container exists on this machine

From the **host**:

```bash
docker ps -a --filter name=mochi
```

- **No output** → the `mochi` container does not exist on this machine. Create it with **Create and enter the container** below (and install deps inside).
- **One line** (e.g. `... mochi`) → the container exists. If status is **Exited**, start and attach with `docker start -ai mochi`. If status is **Up**, attach with `docker exec -it mochi bash`.

To list all containers (including other names): `docker ps -a`.

### Create and enter the container

From the **host** (on the machine where you will run Mochi). Use this only if the `mochi` container does not exist (see **Check if the Mochi container exists** above):

```bash
docker run -it --gpus all --name mochi \
  -v /nfs/oagrawal/mochi:/workspace/mochi \
  pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel bash
```

Adjust `-v` if your Mochi path differs. Inside the container, the repo is at `/workspace/mochi`. If the repo is on a shared mount (e.g. NFS), the same `docker run` creates the container on this host; the image may need to be pulled first: `docker pull pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel`.

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

In `teacache_mochi.py`, use either **`--teacache_thresh`** (single value, default `0.09`) or **`--teacache_thresh_low`** and **`--teacache_thresh_high`**. With low/high: same value = fixed threshold for all steps; different values = adaptive (first 33 steps use high, rest use low). Single threshold trades off latency vs quality:

- **`0.06`** — ~1.5× speedup, stricter (better quality, less skip)
- **`0.09`** — ~2.1× speedup (default when using `--teacache_thresh`)

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

**Mode:** Only `mochi_baseline` is supported by the batch script. The batch script (`batch_generate_mochi.py`) uses the native Mochi pipeline (`MochiSingleGPUPipeline`), which doesn't include TeaCache. TeaCache is implemented separately in `teacache_mochi.py` using the diffusers pipeline, but it's not integrated into the batch script. To use TeaCache in batch mode, you would need to modify the batch script to use the diffusers pipeline (see `teacache_mochi.py` for reference).

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

**Split across 4 GPUs:**

For parallel execution across 4 GPUs, split the 33 prompts as follows:

- **GPU 0**: prompts 0-8 (9 prompts) → `--start-idx 0 --end-idx 9`
- **GPU 1**: prompts 9-17 (9 prompts) → `--start-idx 9 --end-idx 18`
- **GPU 2**: prompts 18-26 (9 prompts) → `--start-idx 18 --end-idx 27`
- **GPU 3**: prompts 27-32 (6 prompts) → `--start-idx 27 --end-idx 33`

**Race condition safety:** The batch script is designed for parallel execution:
- ✅ Separate log files per GPU range (`generation_log_{start_idx}-{end_idx}.json`)
- ✅ Atomic log writes (tmp file + `os.replace()`)
- ✅ File existence checks before generation (resume-safe)
- ✅ Unique video filenames (`{prompt}-{seed}.mp4`)
- ✅ Idempotent directory creation

**No additional locking mechanisms needed** - you can start all 4 GPUs simultaneously.

**Setup for 4 GPUs:**

**Terminal 1 (GPU 0):**
```bash
# On host
tmux new -s mochi_gen0

# Inside tmux
docker start -ai mochi
cd /workspace/mochi && source .venv/bin/activate

# Dry run first (verify prompt range)
python3 vbench_eval/batch_generate_mochi.py \
  --model_dir weights/ --cpu_offload \
  --output-dir vbench_eval/videos \
  --start-idx 0 --end-idx 9 \
  --dry-run

# Actual generation
CUDA_VISIBLE_DEVICES=0 python3 vbench_eval/batch_generate_mochi.py \
  --model_dir weights/ --cpu_offload \
  --output-dir vbench_eval/videos \
  --start-idx 0 --end-idx 9
```

**Terminal 2 (GPU 1):**
```bash
# On host
tmux new -s mochi_gen1

# Inside tmux
docker start -ai mochi
cd /workspace/mochi && source .venv/bin/activate

# Dry run first
python3 vbench_eval/batch_generate_mochi.py \
  --model_dir weights/ --cpu_offload \
  --output-dir vbench_eval/videos \
  --start-idx 9 --end-idx 18 \
  --dry-run

# Actual generation
CUDA_VISIBLE_DEVICES=1 python3 vbench_eval/batch_generate_mochi.py \
  --model_dir weights/ --cpu_offload \
  --output-dir vbench_eval/videos \
  --start-idx 9 --end-idx 18
```

**Terminal 3 (GPU 2):**
```bash
# On host
tmux new -s mochi_gen2

# Inside tmux
docker start -ai mochi
cd /workspace/mochi && source .venv/bin/activate

# Dry run first
python3 vbench_eval/batch_generate_mochi.py \
  --model_dir weights/ --cpu_offload \
  --output-dir vbench_eval/videos \
  --start-idx 18 --end-idx 27 \
  --dry-run

# Actual generation
CUDA_VISIBLE_DEVICES=2 python3 vbench_eval/batch_generate_mochi.py \
  --model_dir weights/ --cpu_offload \
  --output-dir vbench_eval/videos \
  --start-idx 18 --end-idx 27
```

**Terminal 4 (GPU 3):**
```bash
# On host
tmux new -s mochi_gen3

# Inside tmux
docker start -ai mochi
cd /workspace/mochi && source .venv/bin/activate

# Dry run first
python3 vbench_eval/batch_generate_mochi.py \
  --model_dir weights/ --cpu_offload \
  --output-dir vbench_eval/videos \
  --start-idx 27 --end-idx 33 \
  --dry-run

# Actual generation
CUDA_VISIBLE_DEVICES=3 python3 vbench_eval/batch_generate_mochi.py \
  --model_dir weights/ --cpu_offload \
  --output-dir vbench_eval/videos \
  --start-idx 27 --end-idx 33
```

**Monitoring progress:**

```bash
# Count completed videos (should reach 33)
ls -1 /nfs/oagrawal/mochi/vbench_eval/videos/mochi_baseline/*.mp4 | wc -l

# Check log files
ls -lh /nfs/oagrawal/mochi/vbench_eval/videos/generation_log_*.json

# View a specific log (requires jq)
cat /nfs/oagrawal/mochi/vbench_eval/videos/generation_log_0-9.json | jq '.runs | length'
```

**Reattach to tmux sessions** (if SSH disconnects):
```bash
tmux attach -t mochi_gen0  # GPU 0
tmux attach -t mochi_gen1  # GPU 1
tmux attach -t mochi_gen2  # GPU 2
tmux attach -t mochi_gen3  # GPU 3
tmux ls  # List all sessions
```

**Recovery:** If a process crashes or is interrupted, simply re-run the same command - the script automatically skips existing videos and resumes from where it left off. Each GPU can be restarted independently.

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
