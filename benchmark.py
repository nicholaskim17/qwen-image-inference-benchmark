#!/usr/bin/env python3
"""Benchmark Qwen-Image latency while changing only inference step count."""

import argparse
import csv
import platform
import time
from pathlib import Path

from mflux.models.common.config import ModelConfig
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage


DEFAULT_PROMPT = (
    "A cinematic photo of a futuristic red sports car driving through downtown "
    "Toronto at night, wet streets reflecting neon lights, detailed buildings, "
    "realistic photography"
)
DEFAULT_MODEL = "mlx-community/Qwen-Image-2512-8bit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, nargs="+", default=[8, 15, 25, 40])
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Skip the untimed 1-step run used to warm MLX and the prompt cache.",
    )
    return parser.parse_args()


def print_summary(rows: list[dict[str, object]]) -> None:
    print("\nSummary")
    print(f"{'Steps':>5}  {'Runtime (s)':>11}  Output")
    print(f"{'-' * 5}  {'-' * 11}  {'-' * 24}")
    for row in rows:
        print(f"{row['steps']:>5}  {row['runtime_seconds']:>11.2f}  {row['output_file']}")


def main() -> None:
    args = parse_args()
    if any(steps <= 0 for steps in args.steps):
        raise SystemExit("All inference step counts must be positive integers.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir.parent / "results.csv"

    print(f"Loading {args.model}...")
    # This checkpoint already stores 8-bit MLX weights. Leaving quantize=None tells
    # MFlux to honor the checkpoint's stored precision instead of re-quantizing it.
    model = QwenImage(
        model_path=args.model,
        quantize=None,
        model_config=ModelConfig.qwen_image(),
    )

    if not args.skip_warmup:
        print("Warming up with 1 untimed step...")
        model.generate_image(
            seed=args.seed,
            prompt=args.prompt,
            num_inference_steps=1,
            width=args.width,
            height=args.height,
            guidance=args.guidance,
            scheduler="linear",
        )

    rows: list[dict[str, object]] = []
    for steps in args.steps:
        output_path = args.output_dir / f"steps_{steps}.png"
        print(f"\nGenerating {steps} steps...")

        start = time.perf_counter()
        image = model.generate_image(
            seed=args.seed,  # Recreates the same initial latent noise for every run.
            prompt=args.prompt,
            num_inference_steps=steps,
            width=args.width,
            height=args.height,
            guidance=args.guidance,  # Classifier-free guidance strength, held fixed.
            scheduler="linear",  # MFlux's currently supported built-in Qwen scheduler.
        )
        runtime_seconds = time.perf_counter() - start
        image.save(str(output_path))

        row = {
            "steps": steps,
            "runtime_seconds": runtime_seconds,
            "output_file": str(output_path),
            "model": args.model,
            "seed": args.seed,
            "width": args.width,
            "height": args.height,
            "guidance": args.guidance,
            "scheduler": "linear",
            "prompt": args.prompt,
            "hardware": f"{platform.machine()} / {platform.platform()}",
        }
        rows.append(row)

        # Rewrite after each completed run so an interrupted benchmark keeps its data.
        with results_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=row.keys())
            writer.writeheader()
            writer.writerows(rows)

        print(f"Runtime: {runtime_seconds:.2f} sec")
        print(f"Saved: {output_path}")

    print_summary(rows)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
