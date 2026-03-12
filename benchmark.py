"""
Benchmark script for comparing standard vs speculative generation on Qwen3.5.

Usage:
    source ~/Documents/mlx-community/3-11-mlx-community-env/bin/activate
    cd /Users/alex/Documents/Codes/RefSources/mlx-vlm-optimized
    python benchmark.py [--model PATH] [--max-tokens N] [--runs N]
"""

import argparse
import sys
import time

import mlx.core as mx

# Add parent directory for imports
sys.path.insert(0, "/Users/alex/Documents/Codes/RefSources/mlx-vlm-optimized")

from mlx_vlm import load
from mlx_vlm.generate import generate as standard_generate
from speculative_generate import speculative_generate


DEFAULT_MODEL = "/Users/alex/Documents/mlx-community/Qwen3.5-9B-8bit"

# Test prompts of different lengths
TEST_PROMPTS = {
    "short": "Hello, how are you?",
    "medium": (
        "Explain the concept of attention mechanisms in transformer neural networks. "
        "Cover self-attention, multi-head attention, and how they enable the model to "
        "process sequences effectively. Include the mathematical formulation of "
        "scaled dot-product attention."
    ),
    "long": (
        "Write a detailed technical analysis of the following topics:\n\n"
        "1. How does Flash Attention optimize the standard attention mechanism? "
        "Explain the tiling approach, online softmax computation, and memory "
        "complexity reduction from O(N^2) to O(N).\n\n"
        "2. Compare and contrast three approaches to KV cache management: "
        "standard concatenation, rotating/sliding window, and paged attention. "
        "Discuss their trade-offs in terms of memory efficiency, implementation "
        "complexity, and support for variable-length sequences.\n\n"
        "3. Describe speculative decoding: how a small draft model can accelerate "
        "generation from a large model. Explain the verification step, acceptance "
        "criteria, and theoretical speedup bounds.\n\n"
        "4. Analyze the GatedDeltaNet architecture used in Qwen3.5 for linear "
        "attention layers. How does the recurrent state update work? Why is it "
        "efficient for token generation but sequential during prefill?\n\n"
        "Please provide concrete examples and mathematical formulations where appropriate."
    ),
    "summary": (
        "Please summarize the following technical report in bullet points, preserving "
        "all key details, numbers, and technical terms:\n\n"
        "The Apple M2 Ultra chip features a 24-core CPU with 16 performance cores and "
        "8 efficiency cores, paired with a 76-core GPU and 192GB of unified memory with "
        "800 GB/s memory bandwidth. In our benchmark testing of large language model "
        "inference, we compared llama.cpp and MLX frameworks running the Qwen3.5-9B model "
        "with 8-bit quantization.\n\n"
        "For prompt processing (prefill), llama.cpp achieved 955 tokens per second for "
        "128-token prompts and 1167 tokens per second for 512-token prompts. MLX achieved "
        "263 tokens per second for 37-token prompts and 627 tokens per second for 366-token "
        "prompts. This gives llama.cpp a 1.9x to 3.6x advantage in prompt processing.\n\n"
        "For token generation (decode), MLX achieved 65.0 tokens per second while llama.cpp "
        "achieved 42.4 tokens per second. This means MLX is 1.53x faster for token generation. "
        "The memory bandwidth utilization was 85% for MLX versus 63% for llama.cpp.\n\n"
        "The Qwen3.5-9B architecture uses a mixed attention design with 75% linear attention "
        "layers (GatedDeltaNet) and 25% full attention layers. The GatedDeltaNet layers use "
        "a recurrence relation: state_t = g * state_{t-1} + k * delta_t. This recurrence is "
        "purely sequential during prefill but extremely efficient during token generation, "
        "contributing less than 0.2% of per-token compute. The full attention layers use "
        "head_dim=256, which required custom Metal kernel support for fused SDPA.\n\n"
        "Key optimization opportunities identified include: speculative decoding with N-gram "
        "prompt lookup, prefill chunking optimization, KV cache quantization, and custom "
        "Metal shaders for head_dim=256. The speculative decoding approach uses checkpoint "
        "and restore for the ArraysCache (non-trimmable) and offset rollback for KVCache "
        "(trimmable)."
    ),
}


def run_standard_benchmark(model, processor, prompt, max_tokens=256, runs=3):
    """Run standard generation benchmark."""
    results = []
    for i in range(runs):
        mx.clear_cache()
        mx.reset_peak_memory()

        result = standard_generate(
            model,
            processor,
            prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            verbose=False,
        )

        results.append(
            {
                "prompt_tokens": result.prompt_tokens,
                "generation_tokens": result.generation_tokens,
                "prompt_tps": result.prompt_tps,
                "generation_tps": result.generation_tps,
                "peak_memory": result.peak_memory,
            }
        )
        # Skip first run (JIT warmup)
        if i == 0:
            print(f"  [Standard] Run {i+1} (warmup): "
                  f"prompt={result.prompt_tps:.1f} t/s, "
                  f"gen={result.generation_tps:.1f} t/s")
        else:
            print(f"  [Standard] Run {i+1}: "
                  f"prompt={result.prompt_tps:.1f} t/s, "
                  f"gen={result.generation_tps:.1f} t/s")

    # Average over non-warmup runs
    valid = results[1:] if len(results) > 1 else results
    return {
        "prompt_tps": sum(r["prompt_tps"] for r in valid) / len(valid),
        "generation_tps": sum(r["generation_tps"] for r in valid) / len(valid),
        "peak_memory": max(r["peak_memory"] for r in valid),
        "prompt_tokens": valid[0]["prompt_tokens"],
        "generation_tokens": valid[0]["generation_tokens"],
    }


def run_speculative_benchmark(
    model, processor, prompt, max_tokens=256, runs=3,
    ngram_size=3, max_draft_tokens=5
):
    """Run speculative decoding benchmark."""
    results = []
    for i in range(runs):
        mx.clear_cache()
        mx.reset_peak_memory()

        result = speculative_generate(
            model,
            processor,
            prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            verbose=False,
            ngram_size=ngram_size,
            max_draft_tokens=max_draft_tokens,
        )

        results.append(
            {
                "prompt_tokens": result.prompt_tokens,
                "generation_tokens": result.generation_tokens,
                "prompt_tps": result.prompt_tps,
                "generation_tps": result.generation_tps,
                "peak_memory": result.peak_memory,
                "acceptance_rate": result.acceptance_rate,
                "tokens_per_step": result.tokens_per_step,
            }
        )
        if i == 0:
            print(f"  [Speculative] Run {i+1} (warmup): "
                  f"prompt={result.prompt_tps:.1f} t/s, "
                  f"gen={result.generation_tps:.1f} t/s, "
                  f"accept={result.acceptance_rate:.1%}, "
                  f"tok/step={result.tokens_per_step:.2f}")
        else:
            print(f"  [Speculative] Run {i+1}: "
                  f"prompt={result.prompt_tps:.1f} t/s, "
                  f"gen={result.generation_tps:.1f} t/s, "
                  f"accept={result.acceptance_rate:.1%}, "
                  f"tok/step={result.tokens_per_step:.2f}")

    valid = results[1:] if len(results) > 1 else results
    return {
        "prompt_tps": sum(r["prompt_tps"] for r in valid) / len(valid),
        "generation_tps": sum(r["generation_tps"] for r in valid) / len(valid),
        "peak_memory": max(r["peak_memory"] for r in valid),
        "prompt_tokens": valid[0]["prompt_tokens"],
        "generation_tokens": valid[0]["generation_tokens"],
        "acceptance_rate": sum(r["acceptance_rate"] for r in valid) / len(valid),
        "tokens_per_step": sum(r["tokens_per_step"] for r in valid) / len(valid),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark standard vs speculative generation")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model path")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens to generate")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per test")
    parser.add_argument("--ngram-size", type=int, default=3, help="N-gram size for lookup")
    parser.add_argument("--max-draft", type=int, default=5, help="Max draft tokens")
    parser.add_argument("--prompt", type=str, choices=["short", "medium", "long", "summary", "all"],
                        default="all", help="Which prompt to test")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model, processor = load(args.model)
    print(f"Model loaded. Peak memory: {mx.get_peak_memory() / 1e9:.2f} GB\n")

    prompts = TEST_PROMPTS if args.prompt == "all" else {args.prompt: TEST_PROMPTS[args.prompt]}

    all_results = {}
    for prompt_name, prompt_text in prompts.items():
        print(f"\n{'='*60}")
        print(f"Test: {prompt_name} prompt")
        print(f"{'='*60}")

        # Standard generation
        print(f"\nStandard generation (max_tokens={args.max_tokens}):")
        std_result = run_standard_benchmark(
            model, processor, prompt_text,
            max_tokens=args.max_tokens, runs=args.runs
        )

        # Speculative generation
        print(f"\nSpeculative generation (ngram={args.ngram_size}, draft={args.max_draft}):")
        spec_result = run_speculative_benchmark(
            model, processor, prompt_text,
            max_tokens=args.max_tokens, runs=args.runs,
            ngram_size=args.ngram_size, max_draft_tokens=args.max_draft
        )

        # Comparison
        speedup = spec_result["generation_tps"] / std_result["generation_tps"] if std_result["generation_tps"] > 0 else 0
        mem_overhead = spec_result["peak_memory"] - std_result["peak_memory"]

        print(f"\n--- Results: {prompt_name} ---")
        print(f"{'Metric':<25} {'Standard':>12} {'Speculative':>12} {'Delta':>12}")
        print(f"{'-'*61}")
        print(f"{'Prompt (t/s)':<25} {std_result['prompt_tps']:>12.1f} {spec_result['prompt_tps']:>12.1f}")
        print(f"{'Generation (t/s)':<25} {std_result['generation_tps']:>12.1f} {spec_result['generation_tps']:>12.1f} {speedup:>11.2f}x")
        print(f"{'Peak Memory (GB)':<25} {std_result['peak_memory']:>12.2f} {spec_result['peak_memory']:>12.2f} {mem_overhead:>+11.2f}")
        print(f"{'Acceptance Rate':<25} {'N/A':>12} {spec_result['acceptance_rate']:>11.1%}")
        print(f"{'Tokens/Step':<25} {'1.00':>12} {spec_result['tokens_per_step']:>12.2f}")

        all_results[prompt_name] = {
            "standard": std_result,
            "speculative": spec_result,
            "speedup": speedup,
        }

    # Final summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Test':<10} {'Std Gen t/s':>12} {'Spec Gen t/s':>13} {'Speedup':>10} {'Accept%':>10}")
    print(f"{'-'*55}")
    for name, r in all_results.items():
        print(f"{name:<10} {r['standard']['generation_tps']:>12.1f} {r['speculative']['generation_tps']:>13.1f} "
              f"{r['speedup']:>9.2f}x {r['speculative']['acceptance_rate']:>9.1%}")


if __name__ == "__main__":
    main()
