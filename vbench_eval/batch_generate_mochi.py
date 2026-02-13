#!/usr/bin/env python3
"""
Batch video generation for VBench evaluation using Mochi.

Generates videos for each prompt in mochi_baseline mode (no TeaCache).
Saves in VBench naming: {prompt}-{seed}.mp4.
Supports resume (skips existing), --start-idx/--end-idx for GPU splitting.

Usage (from Mochi repo root, inside Mochi Docker/venv):
  CUDA_VISIBLE_DEVICES=0 python3 vbench_eval/batch_generate_mochi.py \\
    --model_dir weights/ --cpu_offload --output-dir vbench_eval/videos
  CUDA_VISIBLE_DEVICES=0 python3 vbench_eval/batch_generate_mochi.py \\
    --model_dir weights/ --cpu_offload --output-dir vbench_eval/videos \\
    --start-idx 0 --end-idx 17
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

# Mochi repo root = parent of vbench_eval
MOCHI_ROOT = str(Path(__file__).resolve().parent.parent)

MODE_NAME = "mochi_baseline"


def load_generation_log(log_path):
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            return json.load(f)
    return {"runs": [], "completed_keys": []}


def save_generation_log(log_path, log_data):
    tmp = log_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(log_data, f, indent=2)
    os.replace(tmp, log_path)


def main():
    parser = argparse.ArgumentParser(description="Mochi VBench batch video generation")
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
        help="Base output directory for videos (mode subdir created here)",
    )
    parser.add_argument("--model_dir", type=str, required=True, help="Path to Mochi weights (e.g. weights/)")
    parser.add_argument("--generation-seed", type=int, default=0)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=-1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cpu_offload", action="store_true", help="Offload model to CPU to reduce VRAM")
    parser.add_argument("--width", type=int, default=848)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--num_frames", type=int, default=163)
    parser.add_argument("--num_steps", type=int, default=64)
    parser.add_argument("--cfg_scale", type=float, default=6.0)
    args = parser.parse_args()

    with open(args.prompts_file, "r") as f:
        all_prompts = json.load(f)

    end_idx = len(all_prompts) if args.end_idx == -1 else args.end_idx
    start_idx = args.start_idx
    prompts = all_prompts[start_idx:end_idx]

    seed = args.generation_seed
    output_dir = os.path.abspath(args.output_dir)
    video_dir = os.path.join(output_dir, MODE_NAME)
    total_videos = len(prompts)

    print("=" * 70)
    print("Mochi VBench Batch Video Generation")
    print("=" * 70)
    print(f"Prompts file:   {args.prompts_file}")
    print(f"Prompt range:   [{start_idx}, {end_idx}) = {len(prompts)} prompts")
    print(f"Seed:            {seed}")
    print(f"Mode:            {MODE_NAME}")
    print(f"Total videos:    {total_videos}")
    print(f"Output dir:     {output_dir}")
    print(f"Model dir:      {args.model_dir}")
    print("=" * 70)

    if args.dry_run:
        print("\n[DRY RUN] Would generate:")
        for entry in prompts:
            prompt = entry["prompt_en"]
            fn = f"{prompt}-{seed}.mp4"
            path = os.path.join(video_dir, fn)
            st = "EXISTS" if os.path.exists(path) else "NEW"
            print(f"  [{st}] {MODE_NAME}/{fn}")
        existing = sum(1 for e in prompts if os.path.exists(os.path.join(video_dir, f"{e['prompt_en']}-{seed}.mp4")))
        print(f"\nAlready exist: {existing}, to generate: {total_videos - existing}")
        return

    log_filename = f"generation_log_{start_idx}-{end_idx}.json"
    log_path = os.path.join(output_dir, log_filename)
    gen_log = load_generation_log(log_path)
    print(f"Log file: {log_path}\n")

    # Import Mochi pipeline (after parsing args so --dry-run exits before loading heavy deps)
    try:
        from genmo.lib.utils import save_video
        from genmo.mochi_preview.pipelines import (
            DecoderModelFactory,
            DitModelFactory,
            MochiSingleGPUPipeline,
            T5ModelFactory,
            linear_quadratic_schedule,
        )
    except ImportError as e:
        print(f"ERROR: Mochi imports failed. Run from Mochi repo root with Mochi env activated: {e}")
        sys.exit(1)

    model_dir = os.path.abspath(args.model_dir)
    dit_path = os.path.join(model_dir, "dit.safetensors")
    decoder_path = os.path.join(model_dir, "decoder.safetensors")
    if not os.path.isfile(dit_path) or not os.path.isfile(decoder_path):
        print(f"ERROR: Weights not found. Expected {dit_path} and {decoder_path}")
        sys.exit(1)

    pipeline = MochiSingleGPUPipeline(
        text_encoder_factory=T5ModelFactory(),
        dit_factory=DitModelFactory(model_path=dit_path, model_dtype="bf16"),
        decoder_factory=DecoderModelFactory(model_path=decoder_path),
        cpu_offload=args.cpu_offload,
        decode_type="tiled_spatial",
    )

    sigma_schedule = linear_quadratic_schedule(args.num_steps, 0.025)
    cfg_schedule = [args.cfg_scale] * args.num_steps

    completed = 0
    skipped = 0
    failed = 0
    total_gen_time = 0.0

    for prompt_idx, entry in enumerate(prompts):
        prompt = entry["prompt_en"]
        global_idx = start_idx + prompt_idx
        video_filename = f"{prompt}-{seed}.mp4"
        video_path = os.path.join(video_dir, video_filename)
        run_num = prompt_idx + 1
        run_key = f"{MODE_NAME}|{prompt}|{seed}"

        if os.path.exists(video_path):
            print(f"[{run_num}/{total_videos}] SKIP (exists): {MODE_NAME} | {prompt[:50]}...")
            skipped += 1
            continue

        print(f"[{run_num}/{total_videos}] Generating: {MODE_NAME} | {prompt[:50]}...")
        os.makedirs(video_dir, exist_ok=True)

        try:
            t0 = time.time()
            final_frames = pipeline(
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=args.num_steps,
                sigma_schedule=sigma_schedule,
                cfg_schedule=cfg_schedule,
                batch_cfg=False,
                prompt=prompt,
                negative_prompt="",
                seed=seed,
            )
            gen_time = time.time() - t0

            import numpy as np
            frames = final_frames[0]
            if hasattr(frames, "cpu"):
                frames = frames.cpu().numpy()
            if not isinstance(frames, np.ndarray):
                frames = np.asarray(frames)
            if frames.dtype != np.float32:
                frames = frames.astype(np.float32)
            save_video(frames, video_path)

            prompt_short = (prompt[:48] + "..") if len(prompt) > 50 else prompt
            print(f"  [{run_num}/{total_videos}] {MODE_NAME:18} | {gen_time:6.1f}s | {prompt_short}")
            print(f"      -> {video_path}")
            completed += 1
            total_gen_time += gen_time
            gen_log["runs"].append({
                "prompt": prompt,
                "seed": seed,
                "mode": MODE_NAME,
                "time_seconds": round(gen_time, 2),
                "video_path": video_path,
                "timestamp": datetime.now().isoformat(),
                "prompt_index": global_idx,
            })
            gen_log["completed_keys"].append(run_key)
            save_generation_log(log_path, gen_log)

        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1
            gen_log["runs"].append({
                "prompt": prompt,
                "seed": seed,
                "mode": MODE_NAME,
                "error": str(e),
                "prompt_index": global_idx,
                "timestamp": datetime.now().isoformat(),
            })
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
