# Mochi VBench Experiment

Same experiment as **Wan2.1** and **HunyuanVideo**: generate videos for the 33 VBench prompts and run the full evaluation pipeline (VBench 16 dimensions + optional fidelity + comparison tables).

## Layout

- **Staging:** `/nfs/oagrawal/mochi_experiment/` — contains this README, `INSTRUCTIONS_MOCHI.md`, and `vbench_eval/` (prompts + batch script).
- **Repo:** Clone [genmoai/mochi](https://github.com/genmoai/mochi) to `/nfs/oagrawal/mochi`, then copy the contents of `mochi_experiment` into it.

## Modes

Currently **one mode**: `mochi_baseline` (no TeaCache for Mochi). If TeaCache adds Mochi support later, you can add `mochi_fixed_0.1`, `mochi_fixed_0.2`, `mochi_adaptive` and extend the batch script and instructions.

## Steps (summary)

1. Clone Mochi → copy `mochi_experiment` into repo → create `weights/`.
2. Run Mochi in Docker (see `INSTRUCTIONS_MOCHI.md` for image and install). Use **uv** for the venv inside the container.
3. Download weights with `scripts/download_weights.py weights/`.
4. **(Optional)** Mini-experiment for TeaCache adaptive threshold: 2 prompts × 4 configs in the Mochi container, then fidelity in the HunyuanVideo eval container. See **Mini-experiment** in **INSTRUCTIONS_MOCHI.md**.
5. Batch generate: `batch_generate_mochi.py --model_dir weights/ --cpu_offload --output-dir vbench_eval/videos` (optionally split with `--start-idx` / `--end-idx`).
6. Run VBench in the HunyuanVideo eval container (same as Wan2.1): `run_vbench_eval.py` with `--video-dir` / `--save-dir` pointing to Mochi’s `vbench_eval`.
7. Run `compare_results.py` with `--modes mochi_baseline` to produce the 3 CSVs.

Full details: **INSTRUCTIONS_MOCHI.md**.
