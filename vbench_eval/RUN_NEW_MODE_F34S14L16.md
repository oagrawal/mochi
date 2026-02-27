# Run mochi_adaptive_f34s14l16 mode (video gen, VBench, fidelity)

This mode uses: **first 34 steps high** (0.12), **steps 34–47 low** (0.04), **last 16 steps high** (0.12). Same prompts, seed 0, and 64 steps as your existing modes.

---

## 0. Containers (ensure they exist)

**Mochi container** (video generation):

```bash
docker ps -a --filter name=mochi
```

- **No output** → create: `docker run -it --gpus all --name mochi -v /nfs/oagrawal/mochi:/workspace/mochi pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel bash` (then install deps inside; see INSTRUCTIONS_MOCHI.md Step 1)
- **Exited** → start: `docker start mochi`
- **Up** → attach: `docker exec -it mochi bash`

**HunyuanVideo eval container** (VBench + fidelity):

```bash
docker ps -a --filter name=hunyuanvideo_eval_wan
```

- **No output** → create with the `docker run` in INSTRUCTIONS_MOCHI.md Step 5
- **Exited** → start: `docker start hunyuanvideo_eval_wan`
- **Up** → attach: `docker exec -it hunyuanvideo_eval_wan bash`

**Tmux:** Run on the **host** (container may not have tmux). Use one tmux session per GPU; run `docker exec -it mochi bash` inside each to get a shell. Detach: `Ctrl+B D`. Reattach: `tmux attach -t mochi_f34_0` (etc).

---

## 1. Video generation (Mochi container, 4 GPUs)

Generate only the new mode (33 prompts × seed 0), split across 4 GPUs.

**Tmux (SSH-safe):** Run tmux on the **host** (not inside the container). Generation takes hours; if SSH disconnects, reattach with `tmux attach -t mochi_f34_0` etc. and the jobs continue. Detach anytime: `Ctrl+B D`.

**On the host:** create 4 tmux sessions (one per GPU):

```bash
tmux new -s mochi_f34_0   # GPU 0
# ... run commands below, then Ctrl+B D to detach ...

tmux new -s mochi_f34_1   # GPU 1
tmux new -s mochi_f34_2   # GPU 2
tmux new -s mochi_f34_3   # GPU 3
```

In each tmux session: attach to the Mochi container, then run the command for that GPU. Use `docker exec -it mochi bash` for extra shells (do **not** use `docker start -ai mochi` in every terminal).

**Terminal 1 (tmux mochi_f34_0, GPU 0, prompts 0–9):**

```bash
docker start mochi
docker exec -it mochi bash

cd /workspace/mochi
source .venv/bin/activate

# Dry run first
python3 vbench_eval/batch_generate_mochi_teacache.py \
  --output-dir vbench_eval/videos \
  --modes mochi_adaptive_f34s14l16 \
  --start-idx 0 --end-idx 9 --dry-run

# Actual generation
CUDA_VISIBLE_DEVICES=0 python3 vbench_eval/batch_generate_mochi_teacache.py \
  --output-dir vbench_eval/videos \
  --modes mochi_adaptive_f34s14l16 \
  --start-idx 0 --end-idx 9
```

**Terminal 2 (tmux mochi_f34_1, GPU 1, prompts 9–18):**

```bash
docker exec -it mochi bash
cd /workspace/mochi
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=1 python3 vbench_eval/batch_generate_mochi_teacache.py \
  --output-dir vbench_eval/videos \
  --modes mochi_adaptive_f34s14l16 \
  --start-idx 9 --end-idx 18
```

**Terminal 3 (tmux mochi_f34_2, GPU 2, prompts 18–27):**

```bash
docker exec -it mochi bash
cd /workspace/mochi
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=2 python3 vbench_eval/batch_generate_mochi_teacache.py \
  --output-dir vbench_eval/videos \
  --modes mochi_adaptive_f34s14l16 \
  --start-idx 18 --end-idx 27
```

**Terminal 4 (tmux mochi_f34_3, GPU 3, prompts 27–33):**

```bash
docker exec -it mochi bash
cd /workspace/mochi
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=3 python3 vbench_eval/batch_generate_mochi_teacache.py \
  --output-dir vbench_eval/videos \
  --modes mochi_adaptive_f34s14l16 \
  --start-idx 27 --end-idx 33
```

**Reattach after SSH disconnect:** `tmux attach -t mochi_f34_0` (or mochi_f34_1, mochi_f34_2, mochi_f34_3). Generation keeps running in the background.

Videos go to `vbench_eval/videos/mochi_adaptive_f34s14l16/*.mp4`.

---

## 2. VBench evaluation (HunyuanVideo eval container)

**Tmux (SSH-safe):** Start tmux on the host, then attach to the eval container inside:

```bash
# On host
tmux new -s vbench_f34

# Inside tmux: attach to eval container
docker start hunyuanvideo_eval_wan
docker exec -it hunyuanvideo_eval_wan bash

MOCHI_VBENCH=/nfs/oagrawal/mochi/vbench_eval
HV_ROOT=/nfs/oagrawal/HunyuanVideo
cd $HV_ROOT

CUDA_VISIBLE_DEVICES=0 python3 vbench_eval/run_vbench_eval.py \
  --video-dir $MOCHI_VBENCH/videos \
  --save-dir $MOCHI_VBENCH/vbench_scores_teacache \
  --full-info $MOCHI_VBENCH/prompts_subset.json \
  --modes mochi_adaptive_f34s14l16
```

**Reattach:** `tmux attach -t vbench_f34`

Output: `vbench_scores_teacache/mochi_adaptive_f34s14l16/`.

---

## 3. Fidelity metrics (HunyuanVideo eval container)

**Tmux (SSH-safe):** Use the same tmux session as VBench, or create a new one:

```bash
# On host (if not already in eval container)
tmux new -s fidelity_f34
docker exec -it hunyuanvideo_eval_wan bash

MOCHI_VBENCH=/nfs/oagrawal/mochi/vbench_eval
HV_ROOT=/nfs/oagrawal/HunyuanVideo
cd $HV_ROOT

CUDA_VISIBLE_DEVICES=0 python3 vbench_eval/run_fidelity_metrics.py \
  --video-dir $MOCHI_VBENCH/videos \
  --baseline mochi_diff_baseline \
  --modes mochi_adaptive_f34s14l16 \
  --save-dir $MOCHI_VBENCH/fidelity_metrics_teacache
```

**Reattach:** `tmux attach -t fidelity_f34`

Output: `fidelity_metrics_teacache/mochi_adaptive_f34s14l16_vs_mochi_diff_baseline.json`.

---

## 4. Compare results (add new mode to tables)

Run compare to regenerate tables including the new mode. No GPU needed; run from host or inside the eval container:

```bash
# From host (HunyuanVideo mounts /nfs/oagrawal)
docker exec -it hunyuanvideo_eval_wan bash

MOCHI_VBENCH=/nfs/oagrawal/mochi/vbench_eval
HV_ROOT=/nfs/oagrawal/HunyuanVideo
cd $HV_ROOT

python3 vbench_eval/compare_results.py \
  --scores-dir $MOCHI_VBENCH/vbench_scores_teacache \
  --fidelity-dir $MOCHI_VBENCH/fidelity_metrics_teacache \
  --gen-log-dir $MOCHI_VBENCH/videos \
  --output-json $MOCHI_VBENCH/all_comparison_results_teacache.json \
  --modes mochi_diff_baseline,mochi_fixed_0.04,mochi_fixed_0.12,mochi_adaptive_0.12_0.04,mochi_adaptive_f34s14l16
```

This updates `vbench_scores_table.csv`, `fidelity_table.csv`, `summary_table.csv` (or similar outputs depending on compare_results behavior).
