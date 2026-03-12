# Demystifying LLM Inference Optimization on Apple Silicon

**Why Speculative Decoding Fails and Where the Real Bottlenecks Lie**

> **Work in Progress** — This research is ongoing. Results and conclusions may be updated as additional optimizations are implemented and validated.

*By [AtomGradient](https://github.com/AtomGradient)*

A systematic study of 9 optimization directions for LLM inference on Apple M2 Ultra using the MLX framework.

## Key Results

| Finding | Detail |
|---------|--------|
| MLX Bandwidth Utilization | **85%** of M2 Ultra's 800 GB/s |
| Speculative Decoding Speedup | **0.85-1.00x** (no improvement) |
| Prefill Bottleneck | Quantized GEMM (**57.6%**), not attention (6.7%) |
| Prior Assumptions Corrected | **5 out of 9** optimization proposals were based on misconceptions |

## Corrected Misconceptions

1. "MLX uses 256+ kernel dispatches" -- Actually uses **1 dispatch** call
2. "Metal shaders are JIT-compiled" -- MLX defaults to **AOT** (mlx.metallib)
3. "SIMD matrix multiply not utilized" -- Already used in **steel_attention** and **steel/gemm**
4. "Pure Transformer batch(5) ~ 5x" -- Actually **1.79x** (bandwidth limited)
5. "Speculative decoding +1.5-2x" -- Actually **0.85-1.00x**

## Hardware

- **Chip**: Apple M2 Ultra (76 GPU cores)
- **Memory**: 192 GB unified, 800 GB/s
- **Models**: Qwen3.5-9B-8bit (hybrid), Qwen3-8B-4bit (pure Transformer)

## Files

| File | Description |
|------|-------------|
| `docs/paper.pdf` | Full research paper |
| `docs/index.html` | Bilingual (EN/ZH) interactive results page |
| `speculative_generate.py` | Speculative decoding for Qwen3.5 (mlx-vlm) |
| `speculative_generate_lm.py` | Speculative decoding for Qwen3 (mlx-lm) |
| `cache_utils.py` | Zero-cost cache checkpoint/restore |
| `benchmark.py` / `benchmark_lm.py` | Standard vs speculative benchmarks |
| `profile_forward.py` / `profile_forward_lm.py` | Forward pass profiling |
| `profile_prefill.py` | Prefill performance profiler |

## Quick Start

```bash
source ~/Documents/mlx-community/3-11-mlx-community-env/bin/activate

# Run benchmark (Qwen3.5)
python benchmark.py --prompt all --max-tokens 256

# Run benchmark (Qwen3)
python benchmark_lm.py --prompt all --max-tokens 128

# Profile forward passes
python profile_forward.py      # Qwen3.5
python profile_forward_lm.py   # Qwen3

# Profile prefill
python profile_prefill.py
```

## Publication

- **Paper**: [Download PDF](https://atomgradient.github.io/mlx-inference-bench/paper.pdf)
- **Website**: [https://atomgradient.github.io/mlx-inference-bench/](https://atomgradient.github.io/mlx-inference-bench/)

## Software Versions

MLX 0.31.0 | mlx-vlm 0.3.12 | mlx-lm 0.30.7 | Python 3.11.13

---

*AtomGradient &middot; March 2026*
