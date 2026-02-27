#!/usr/bin/env python3
"""
Batch video generation for VBench evaluation using Mochi + TeaCache (diffusers pipeline).

Generates videos for each prompt and each TeaCache mode:
  - mochi_diff_baseline      — no TeaCache (diffusers pipeline, baseline for fidelity)
  - mochi_fixed_0.04         — fixed TeaCache threshold 0.04
  - mochi_fixed_0.12         — fixed TeaCache threshold 0.12
  - mochi_adaptive_0.12_0.04 — adaptive: first 34 steps high=0.12, rest low=0.04 (f34l30)
  - mochi_adaptive_f34s14l16 — adaptive: first 34 high, steps 34–47 low, last 16 high (f34s14l16)

Saves videos in VBench naming format: {prompt}-{seed}.mp4
Supports:
  - Resume: skips videos that already exist
  - GPU splitting: --start-idx / --end-idx to split prompts across GPUs
  - Per-process generation log with timing information (JSON)

Usage (from Mochi repo root, inside Mochi Docker/uv venv):

  # Single GPU (all 33 prompts × 4 modes)
  CUDA_VISIBLE_DEVICES=0 python3 vbench_eval/batch_generate_mochi_teacache.py \
    --output-dir vbench_eval/videos

  # 4 GPUs (split prompts 0–9, 9–18, 18–27, 27–33 — same pattern as baseline script):
  # GPU 0
  CUDA_VISIBLE_DEVICES=0 python3 vbench_eval/batch_generate_mochi_teacache.py \
    --output-dir vbench_eval/videos --start-idx 0 --end-idx 9
  # GPU 1
  CUDA_VISIBLE_DEVICES=1 python3 vbench_eval/batch_generate_mochi_teacache.py \
    --output-dir vbench_eval/videos --start-idx 9 --end-idx 18
  # GPU 2
  CUDA_VISIBLE_DEVICES=2 python3 vbench_eval/batch_generate_mochi_teacache.py \
    --output-dir vbench_eval/videos --start-idx 18 --end-idx 27
  # GPU 3
  CUDA_VISIBLE_DEVICES=3 python3 vbench_eval/batch_generate_mochi_teacache.py \
    --output-dir vbench_eval/videos --start-idx 27 --end-idx 33
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

MOCHI_ROOT = str(Path(__file__).resolve().parent.parent)
if MOCHI_ROOT not in sys.path:
    sys.path.insert(0, MOCHI_ROOT)

MODES = [
    {
        "name": "mochi_diff_baseline",
        "enable_teacache": False,
        "thresh_low": None,
        "thresh_high": None,
    },
    {
        "name": "mochi_fixed_0.04",
        "enable_teacache": True,
        "thresh_low": 0.04,
        "thresh_high": 0.04,
    },
    {
        "name": "mochi_fixed_0.12",
        "enable_teacache": True,
        "thresh_low": 0.12,
        "thresh_high": 0.12,
    },
    {
        "name": "mochi_adaptive_0.12_0.04",
        "enable_teacache": True,
        "thresh_low": 0.04,
        "thresh_high": 0.12,
    },
    {
        "name": "mochi_adaptive_f34s14l16",
        "enable_teacache": True,
        "thresh_low": 0.04,
        "thresh_high": 0.12,
        "adaptive_high_steps": 33,
        "adaptive_back_to_high_step": 48,
    },
]


def load_generation_log(log_path: str):
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            return json.load(f)
    return {"runs": [], "completed_keys": []}


def save_generation_log(log_path: str, log_data):
    tmp = log_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(log_data, f, indent=2)
    os.replace(tmp, log_path)


def main():
    parser = argparse.ArgumentParser(description="Mochi + TeaCache VBench batch video generation (diffusers)")
    parser.add_argument(
        "--prompts-file",
        type=str,
        default=os.path.join(MOCHI_ROOT, "vbench_eval", "prompts_subset.json"),
        help="Path to VBench prompts JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(MOCHI_ROOT, "vbench_eval", "videos"),
        help="Base output directory for videos (mode subdirs created here)",
    )
    parser.add_argument(
        "--generation-seed",
        type=int,
        default=0,
        help="Seed for video generation (keep fixed across modes for fidelity)",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Start prompt index (inclusive, for GPU splitting)",
    )
    parser.add_argument(
        "--end-idx",
        type=int,
        default=-1,
        help="End prompt index (exclusive, -1 = all prompts)",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="all",
        help=        "Comma-separated modes to run, or 'all'. "
             "Options: mochi_diff_baseline,mochi_fixed_0.04,mochi_fixed_0.12,mochi_adaptive_0.12_0.04,mochi_adaptive_f34s14l16",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=64,
        help="Number of inference steps (must match mini-experiment / teacache_mochi defaults)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print what would be generated without running")
    args = parser.parse_args()

    with open(args.prompts_file, "r") as f:
        all_prompts = json.load(f)

    end_idx = len(all_prompts) if args.end_idx == -1 else args.end_idx
    start_idx = args.start_idx
    prompts = all_prompts[start_idx:end_idx]

    # Filter modes
    if args.modes == "all":
        modes = MODES
    else:
        mode_names = [m.strip() for m in args.modes.split(",")]
        modes = [m for m in MODES if m["name"] in mode_names]
        if not modes:
            print(f"ERROR: No valid modes found in '{args.modes}'")
            print(f"Available: {[m['name'] for m in MODES]}")
            sys.exit(1)

    seed = args.generation_seed
    output_dir = os.path.abspath(args.output_dir)
    total_videos = len(prompts) * len(modes)

    print("=" * 70)
    print("Mochi + TeaCache VBench Batch Video Generation (diffusers pipeline)")
    print("=" * 70)
    print(f"Prompts file:  {args.prompts_file}")
    print(f"Prompt range:  [{start_idx}, {end_idx}) = {len(prompts)} prompts")
    print(f"Seed:          {seed}")
    print(f"Modes:         {[m['name'] for m in modes]}")
    print(f"Total videos:  {total_videos}")
    print(f"Output dir:    {output_dir}")
    print("=" * 70)

    # Dry run
    if args.dry_run:
        print("\n[DRY RUN] Would generate these videos:\n")
        for entry in prompts:
            prompt = entry["prompt_en"]
            for mode in modes:
                filename = f"{prompt}-{seed}.mp4"
                filepath = os.path.join(output_dir, mode["name"], filename)
                exists = os.path.exists(filepath)
                status = "EXISTS" if exists else "NEW"
                print(f"  [{status}] {mode['name']}/{filename}")
        existing = sum(
            1
            for entry in prompts
            for mode in modes
            if os.path.exists(os.path.join(output_dir, mode["name"], f"{entry['prompt_en']}-{seed}.mp4"))
        )
        print(f"\nTotal: {total_videos} videos")
        print(f"Already exist: {existing}")
        print(f"To generate: {total_videos - existing}")
        return

    # Import teacache_mochi only for actual generation
    try:
        from teacache_mochi import run_generation
    except ImportError as e:
        print(f"ERROR: Failed to import teacache_mochi.run_generation. "
              f"Run from Mochi repo root with the Mochi env activated: {e}")
        sys.exit(1)

    # Per-process generation log (safe with multi-GPU when start/end ranges don't overlap)
    log_filename = f"generation_log_teacache_{start_idx}-{end_idx}.json"
    log_path = os.path.join(output_dir, log_filename)
    gen_log = load_generation_log(log_path)
    print(f"Log file:      {log_path}")

    completed = 0
    skipped = 0
    failed = 0
    total_gen_time = 0.0

    for prompt_idx, entry in enumerate(prompts):
        prompt = entry["prompt_en"]
        global_idx = start_idx + prompt_idx

        for mode_idx, mode in enumerate(modes):
            mode_name = mode["name"]
            video_filename = f"{prompt}-{seed}.mp4"
            video_dir = os.path.join(output_dir, mode_name)
            video_path = os.path.join(video_dir, video_filename)
            run_num = prompt_idx * len(modes) + mode_idx + 1
            run_key = f"{mode_name}|{prompt}|{seed}"

            # Resume: skip if file already exists
            if os.path.exists(video_path):
                print(f"[{run_num}/{total_videos}] SKIP (exists): {mode_name} | {prompt[:50]}...")
                skipped += 1
                gen_log["completed_keys"].append(run_key)
                save_generation_log(log_path, gen_log)
                continue

            os.makedirs(video_dir, exist_ok=True)
            print(f"[{run_num}/{total_videos}] Generating: {mode_name} | {prompt[:50]}...")

            try:
                t0 = time.time()
                if mode["enable_teacache"]:
                    # Fixed or adaptive TeaCache
                    gen_kwargs = dict(
                        prompt=prompt,
                        num_inference_steps=args.num_steps,
                        seed=seed,
                        out_dir=video_dir,
                        enable_teacache=True,
                        rel_l1_thresh_low=mode["thresh_low"],
                        rel_l1_thresh_high=mode["thresh_high"],
                        save_file=video_path,
                    )
                    gen_kwargs["adaptive_high_steps"] = mode.get("adaptive_high_steps", 33)
                    if "adaptive_back_to_high_step" in mode:
                        gen_kwargs["adaptive_back_to_high_step"] = mode["adaptive_back_to_high_step"]
                    run_generation(**gen_kwargs)
                else:
                    # Diffusers baseline: TeaCache disabled
                    run_generation(
                        prompt=prompt,
                        num_inference_steps=args.num_steps,
                        seed=seed,
                        out_dir=video_dir,
                        enable_teacache=False,
                        rel_l1_thresh=0.09,
                        save_file=video_path,
                    )
                gen_time = time.time() - t0

                prompt_short = (prompt[:48] + "..") if len(prompt) > 50 else prompt
                print(f"  {mode_name:24} | {gen_time:7.1f}s | {prompt_short}")
                print(f"      -> {video_path}")

                completed += 1
                total_gen_time += gen_time
                gen_log["runs"].append(
                    {
                        "prompt": prompt,
                        "prompt_index": global_idx,
                        "seed": seed,
                        "mode": mode_name,
                        "time_seconds": round(gen_time, 1),
                        "video_path": video_path,
                        "timestamp": datetime.now().isoformat(),
                    }
                )
                gen_log["completed_keys"].append(run_key)
                save_generation_log(log_path, gen_log)

            except Exception as e:  # noqa: BLE001
                print(f"  FAILED: {e}")
                failed += 1
                gen_log["runs"].append(
                    {
                        "prompt": prompt,
                        "prompt_index": global_idx,
                        "seed": seed,
                        "mode": mode_name,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat(),
                    }
                )
                save_generation_log(log_path, gen_log)

    print("\n" + "=" * 70)
    print("BATCH GENERATION COMPLETE")
    print("=" * 70)
    print(f"  Completed:     {completed}")
    print(f"  Skipped:       {skipped} (already existed)")
    print(f"  Failed:        {failed}")
    if completed:
        print(f"  Total time:    {total_gen_time:.1f}s  ({total_gen_time/3600:.1f}h)")
        print(f"  Avg per video: {total_gen_time/completed:.1f}s")
    print(f"  Log file:      {log_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()

