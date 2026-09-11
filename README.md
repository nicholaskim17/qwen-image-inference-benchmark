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

This script loads the model once, performs one untimed 2-step warm-up, and then
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

Prompt:

> A cinematic photo of a futuristic red sports car driving through downtown
> Toronto at night, wet streets reflecting neon lights, detailed buildings,
> realistic photography

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
| 8 | 22.47 | [steps_8.png](outputs/steps_8.png) |
| 15 | 37.71 | [steps_15.png](outputs/steps_15.png) |
| 25 | 82.25 | [steps_25.png](outputs/steps_25.png) |

Measured on September 11, 2026. These are single-run wall-clock observations
after the warm-up, not averages over repeated trials.

## Generated images

### 8 steps

![Qwen-Image output at 8 inference steps](outputs/steps_8.png)

### 15 steps

![Qwen-Image output at 15 inference steps](outputs/steps_15.png)

### 25 steps

![Qwen-Image output at 25 inference steps](outputs/steps_25.png)

## Observations

1. Latency increased substantially with step count. The 15-step run took about
   68% longer than 8 steps, while 25 steps took about 266% longer than 8 steps.
   The 25-step run also showed a brief late-run slowdown, so the relationship was
   not perfectly linear in this single trial.
2. The main composition was already coherent at 8 steps: the car shape, wet
   street, lighting, and buildings were all recognizable. At 15 and 25 steps,
   reflections and body contours appeared somewhat cleaner and more developed,
   but the gain was subtle compared with the extra runtime.
3. Fine details such as signs, windows, and reflections changed across step
   counts even with the same seed. A fixed seed fixes the initial noise, but a
   different number of solver updates follows a different numerical trajectory.

On this hardware and this test prompt, reducing inference steps substantially
reduced latency. Beyond 15 steps, the visible quality improvement appeared
relatively small compared with the additional runtime. This is a prompt-specific
observation, not a claim that 15 steps is generally optimal.

## Limitations

This is a small exploratory benchmark, not a comprehensive evaluation. It uses
one prompt, one seed, one resolution, one quantized checkpoint, and one machine.
Visible quality is assessed informally rather than with human ratings or an
automated metric. Timing is a single observation per condition rather than a
distribution from repeated runs, and background system load is uncontrolled. No
result here establishes a universally optimal inference-step count.
