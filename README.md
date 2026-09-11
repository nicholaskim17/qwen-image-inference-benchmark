# Qwen-Image Inference-Step Benchmark

A small exploratory benchmark of the latency and visible-quality tradeoff from
changing Qwen-Image's inference step count. It uses one prompt and one seed so
the images are directly comparable.

## Experiment

Diffusion and flow-matching image generators start from noise and iteratively
refine a latent representation. More inference steps give the numerical solver
more opportunities to refine the result, but each step adds transformer work and
therefore latency. More steps do not guarantee a proportional visible-quality
gain.

This script loads the model once, performs one untimed 1-step warm-up, and then
generates images at 8, 15, and 25 steps. The warm-up populates MFlux's prompt
cache and triggers MLX's lazy compilation before measurements begin. Wall-clock
timing covers iterative generation and VAE decoding. It excludes model
download/load, the warm-up, and image file writing.

## Configuration

- Hardware: Apple Silicon MacBook Pro, M5 Pro, 48 GB unified memory
- Runtime: MFlux 0.19.1 on MLX 0.32.2
- Model: `mlx-community/Qwen-Image-2512-8bit`
- Quantization: stored 8-bit MLX checkpoint, fixed for every run
- Scheduler: MFlux linear scheduler
- Guidance scale: 3.5
- Resolution: 512 × 320
- Seed: 42

The checkpoint is quantized so the 20B-parameter model is practical on this
machine. Quantization can affect quality, but it is held constant and is not a
benchmark variable. The fixed seed recreates the same initial latent noise. The
prompt, resolution, model, guidance, scheduler, runtime, and hardware are also
held constant. Only `num_inference_steps` changes.

## Run it

Use the existing environment:

```bash
/Users/nicholaskim/qwen-env/bin/python benchmark.py
```

To choose a different sweep explicitly:

```bash
/Users/nicholaskim/qwen-env/bin/python benchmark.py --steps 8 15 25 40
```

Images are written to `outputs/`, and measurements are written to
`results.csv` after every completed generation.

## Results

| Steps | Runtime (seconds) | Image |
| ---: | ---: | --- |
| 8 | Pending | [steps_8.png](outputs/steps_8.png) |
| 15 | Pending | [steps_15.png](outputs/steps_15.png) |
| 25 | Pending | [steps_25.png](outputs/steps_25.png) |

## Generated images

Images will be embedded here after the benchmark completes.

## Observations

Observations will be recorded after inspecting the generated images and timing
data. The intended comparison is whether added latency at higher step counts
corresponds to visible improvements in composition, fine detail, reflections,
and structural coherence for this prompt.

## Limitations

This is a small exploratory benchmark, not a comprehensive evaluation. It uses
one prompt, one seed, one resolution, one quantized checkpoint, and one machine.
Visible quality is assessed informally rather than with human ratings or an
automated metric. Timing is a single observation per condition rather than a
distribution from repeated runs, and background system load is uncontrolled. No
result here establishes a universally optimal inference-step count.
