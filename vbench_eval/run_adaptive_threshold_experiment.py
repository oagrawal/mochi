#!/usr/bin/env python3
"""
Mini-experiment: generate 2 prompts × (1 baseline + 4 adaptive configs) with fixed seed
for TeaCache threshold selection. Outputs to videos_threshold_experiment/ for fidelity eval.

Adaptive = first 33 steps use high threshold, rest use low. Same seed for all runs.

Splitting across 4 GPUs (runtime-balanced):
  Use --gpu-id 0..3 and --total-gpus 4. Baselines are on different GPUs; each baseline
  GPU also runs an aggressive cache config (c2/c4) so runtime is balanced. Job order:
  0=base p0, 1=base p1, 2=c1 p0, 3=c1 p1, 4=c2 p0, 5=c2 p1, 6=c3 p0, 7=c3 p1, 8=c4 p0, 9=c4 p1.

Usage (from Mochi repo root, inside Mochi Docker/venv):
  CUDA_VISIBLE_DEVICES=0 python3 vbench_eval/run_adaptive_threshold_experiment.py --gpu-id 0 --total-gpus 4
  # Or manual range: --start-idx 0 --end-idx 10
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

MOCHI_ROOT = str(Path(__file__).resolve().parent.parent)
if MOCHI_ROOT not in sys.path:
    sys.path.insert(0, MOCHI_ROOT)

# Configs: (thresh_high, thresh_low). First 33 steps use high, rest use low.
ADAPTIVE_CONFIGS = [
    ("mochi_adaptive_c1", 0.08, 0.04),
    ("mochi_adaptive_c2", 0.20, 0.10),
    ("mochi_adaptive_c3", 0.12, 0.04),
    ("mochi_adaptive_c4", 0.16, 0.10),
]
BASELINE_MODE = "mochi_baseline"
DEFAULT_SEED = 0
NUM_PROMPTS = 2
ADAPTIVE_HIGH_STEPS = 33

# Runtime-balanced 4-GPU split: no GPU runs both baselines; each baseline GPU runs an aggressive (c2/c4) job.
# Job indices: 0=base p0, 1=base p1, 2=c1 p0, 3=c1 p1, 4=c2 p0, 5=c2 p1, 6=c3 p0, 7=c3 p1, 8=c4 p0, 9=c4 p1.
# Aggressive (faster): c2, c4. Conservative (slower): c1, c3.
JOB_INDICES_FOR_4GPU = [
    [0, 4, 7],   # GPU 0: baseline p0 + c2 p0 + c3 p1
    [1, 8],      # GPU 1: baseline p1 + c4 p0
    [5, 9, 2],   # GPU 2: c2 p1 + c4 p1 + c1 p0
    [3, 6],      # GPU 3: c1 p1 + c3 p0
]


def build_job_list(prompts):
    """Flat list of (mode_name, prompt, thresh_high|None, thresh_low|None). Baseline first (2), then c1..c4 (2 each) = 10 jobs."""
    jobs = []
    for prompt in prompts:
        jobs.append((BASELINE_MODE, prompt, None, None))  # baseline: no thresh (enable_teacache=False)
    for mode_name, thresh_high, thresh_low in ADAPTIVE_CONFIGS:
        for prompt in prompts:
            jobs.append((mode_name, prompt, thresh_high, thresh_low))
    return jobs


def main():
    parser = argparse.ArgumentParser(description="Mochi TeaCache adaptive threshold mini-experiment")
    parser.add_argument("--prompts-file", type=str, default=os.path.join(MOCHI_ROOT, "vbench_eval", "prompts_subset.json"))
    parser.add_argument("--output-dir", type=str, default=os.path.join(MOCHI_ROOT, "vbench_eval", "videos_threshold_experiment"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--gpu-id", type=int, default=None, help="GPU index (0..total_gpus-1). With --total-gpus 4, uses runtime-balanced job split.")
    parser.add_argument("--total-gpus", type=int, default=None, help="Number of GPUs; use with --gpu-id for balanced 4-GPU split.")
    parser.add_argument("--start-idx", type=int, default=None, help="Start job index (inclusive). Ignored if --gpu-id set.")
    parser.add_argument("--end-idx", type=int, default=None, help="End job index (exclusive). Ignored if --gpu-id set.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with open(args.prompts_file, "r") as f:
        all_prompts = json.load(f)
    prompts = [entry["prompt_en"] for entry in all_prompts[:NUM_PROMPTS]]
    jobs = build_job_list(prompts)
    total_jobs = len(jobs)

    if args.gpu_id is not None and args.total_gpus is not None:
        if args.total_gpus != 4 or args.gpu_id < 0 or args.gpu_id >= 4:
            raise SystemExit("--total-gpus must be 4 and --gpu-id in 0..3 for the predefined balanced split.")
        job_indices = JOB_INDICES_FOR_4GPU[args.gpu_id]
        jobs_slice = [jobs[i] for i in job_indices]
        range_desc = f"gpu-id {args.gpu_id}/{args.total_gpus} (jobs {job_indices})"
    else:
        start_idx = max(0, args.start_idx if args.start_idx is not None else 0)
        end_idx = min(total_jobs, args.end_idx if args.end_idx is not None else total_jobs)
        jobs_slice = jobs[start_idx:end_idx]
        range_desc = f"jobs [{start_idx}, {end_idx})"

    output_dir = os.path.abspath(args.output_dir)
    seed = args.seed

    print("=" * 70)
    print("Mochi TeaCache adaptive threshold mini-experiment")
    print("=" * 70)
    print(f"Prompts:      {NUM_PROMPTS} (first from prompts_subset.json)")
    print(f"Seed:         {seed}")
    print(f"Output dir:   {output_dir}")
    print(f"Job range:    {range_desc} ({len(jobs_slice)} on this process)")
    print("=" * 70)

    if args.dry_run:
        print("\n[DRY RUN] Would generate:")
        for i, (mode_name, prompt, _h, _l) in enumerate(jobs_slice):
            print(f"  {mode_name}/{prompt}-{seed}.mp4")
        print(f"\nTotal this process: {len(jobs_slice)} videos")
        return

    from teacache_mochi import run_generation

    n_here = len(jobs_slice)
    for i, (mode_name, prompt, thresh_high, thresh_low) in enumerate(jobs_slice):
        video_dir = os.path.join(output_dir, mode_name)
        os.makedirs(video_dir, exist_ok=True)
        save_path = os.path.join(video_dir, f"{prompt}-{seed}.mp4")
        job_num = i + 1
        n_here = len(jobs_slice)
        if os.path.exists(save_path):
            print(f"[{job_num}/{n_here}] SKIP (exists): {mode_name} | {prompt[:50]}...")
            continue
        if mode_name == BASELINE_MODE:
            print(f"[{job_num}/{n_here}] Generating baseline: {prompt[:50]}...")
            t0 = time.time()
            run_generation(
                prompt=prompt,
                num_inference_steps=64,
                seed=seed,
                out_dir=video_dir,
                enable_teacache=False,
                rel_l1_thresh=0.09,
                save_file=save_path,
            )
        else:
            print(f"[{job_num}/{n_here}] Generating {mode_name} (high={thresh_high}, low={thresh_low}): {prompt[:50]}...")
            t0 = time.time()
            run_generation(
                prompt=prompt,
                num_inference_steps=64,
                seed=seed,
                out_dir=video_dir,
                enable_teacache=True,
                rel_l1_thresh_low=thresh_low,
                rel_l1_thresh_high=thresh_high,
                adaptive_high_steps=ADAPTIVE_HIGH_STEPS,
                save_file=save_path,
            )
        print(f"  {time.time() - t0:.1f}s -> {save_path}")

    print("\n" + "=" * 70)
    print("Mini-experiment generation complete. Run fidelity in HunyuanVideo eval container.")
    print("=" * 70)


if __name__ == "__main__":
    main()
