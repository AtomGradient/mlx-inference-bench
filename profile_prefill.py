"""
Prefill Performance Profiler for MLX models.

Measures prefill throughput at different prompt lengths, separates
GatedDeltaNet vs Transformer attention costs, and compares Qwen3.5 (hybrid)
vs Qwen3 (pure Transformer).

Usage:
    source ~/Documents/mlx-community/3-11-mlx-community-env/bin/activate
    cd /Users/alex/Documents/Codes/RefSources/mlx-vlm-optimized
    python profile_prefill.py
"""

import sys
import time
import json
import gc

sys.path.insert(0, "/Users/alex/Documents/Codes/RefSources/mlx-vlm-optimized")

import mlx.core as mx
import mlx.nn as nn

QWEN35_PATH = "/Users/alex/Documents/mlx-community/Qwen3.5-9B-8bit"
QWEN3_PATH = "/Users/alex/Documents/mlx-community/Qwen3-8B-4bit"

PROMPT_LENGTHS = [64, 128, 256, 512, 1024, 2048]
NUM_WARMUP = 2
NUM_RUNS = 5


def load_model_and_tokenizer(model_path):
    """Load model using mlx-vlm or mlx-lm depending on availability."""
    # Check config to decide which loader to use
    config_path = f"{model_path}/config.json"
    try:
        with open(config_path) as f:
            config = json.load(f)
        model_type = config.get("model_type", "")
    except Exception:
        model_type = ""

    # Try mlx-vlm first for VLM models
    if model_type in ("qwen3_5",):
        try:
            from mlx_vlm import load as vlm_load
            model, processor = vlm_load(model_path)
            tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
            return model, tokenizer, "vlm"
        except Exception as e:
            print(f"  vlm load failed: {e}")

    # Try mlx-lm for pure LLM models
    try:
        from mlx_lm import load as lm_load
        model, tokenizer = lm_load(model_path)
        return model, tokenizer, "lm"
    except Exception as e:
        print(f"  lm load failed: {e}")

    # Fallback: try vlm anyway
    try:
        from mlx_vlm import load as vlm_load
        model, processor = vlm_load(model_path)
        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        return model, tokenizer, "vlm"
    except Exception as e:
        print(f"  vlm fallback failed: {e}")

    raise RuntimeError(f"Cannot load model from {model_path}")


def get_language_model(model, loader_type):
    """Get the language model component."""
    if loader_type == "vlm" and hasattr(model, "language_model"):
        return model.language_model
    return model


def make_cache(lm_model):
    """Create prompt cache for the model."""
    if hasattr(lm_model, "make_cache"):
        return lm_model.make_cache()
    try:
        from mlx_vlm.models import cache
        return cache.make_prompt_cache(lm_model)
    except Exception:
        pass
    try:
        from mlx_lm.models import cache
        return cache.make_prompt_cache(lm_model)
    except Exception:
        pass
    raise RuntimeError("Cannot create prompt cache")


def eval_cache_safe(prompt_cache):
    """Safely eval cache state, handling uninitialized caches."""
    states = []
    for c in prompt_cache:
        try:
            s = c.state
            if s is not None:
                if isinstance(s, (list, tuple)):
                    states.extend([x for x in s if x is not None])
                else:
                    states.append(s)
        except (AttributeError, TypeError):
            pass
    if states:
        mx.eval(states)


def reset_model_state(lm_model):
    """Reset cached position_ids and rope_deltas (needed for Qwen3.5)."""
    if hasattr(lm_model, "_position_ids"):
        lm_model._position_ids = None
    if hasattr(lm_model, "_rope_deltas"):
        lm_model._rope_deltas = None


def generate_tokens(tokenizer, length):
    """Generate a token sequence of a given length."""
    # Repeat a simple text to fill the desired length
    base_text = "The quick brown fox jumps over the lazy dog. "
    text = base_text * (length // 5 + 1)
    tokens = tokenizer.encode(text)
    if len(tokens) < length:
        tokens = tokens * (length // len(tokens) + 1)
    return tokens[:length]


def get_logits(outputs):
    """Extract logits from model output (handles both named tuple and raw array)."""
    if hasattr(outputs, "logits"):
        return outputs.logits
    return outputs


def profile_prefill(lm_model, tokenizer, prompt_lengths, model_name, loader_type):
    """Profile prefill performance at different prompt lengths."""
    print(f"\n{'='*70}")
    print(f"Prefill Profile: {model_name}")
    print(f"{'='*70}")

    results = []

    for target_len in prompt_lengths:
        tokens = generate_tokens(tokenizer, target_len)
        input_ids = mx.array([tokens])
        actual_len = len(tokens)

        times = []

        for run in range(NUM_WARMUP + NUM_RUNS):
            # Fresh cache each run
            prompt_cache = make_cache(lm_model)
            reset_model_state(lm_model)
            gc.collect()
            mx.clear_cache()

            mx.synchronize()
            t0 = time.perf_counter()

            # Run full prefill (no chunking, to measure raw throughput)
            outputs = lm_model(input_ids, cache=prompt_cache)
            logits = get_logits(outputs)[:, -1, :]
            mx.eval(logits)

            t1 = time.perf_counter()
            elapsed = t1 - t0

            if run >= NUM_WARMUP:
                times.append(elapsed)

        avg_time = sum(times) / len(times)
        std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
        tps = actual_len / avg_time

        print(f"  {actual_len:>5} tokens: {avg_time*1000:>8.1f} ms  "
              f"(+/- {std_time*1000:.1f} ms)  "
              f"{tps:>8.1f} t/s")

        results.append({
            "length": actual_len,
            "time_ms": avg_time * 1000,
            "std_ms": std_time * 1000,
            "tps": tps,
        })

    return results


def profile_chunked_prefill(lm_model, tokenizer, prompt_length, chunk_sizes, model_name):
    """Profile prefill with different chunk sizes."""
    print(f"\n{'='*70}")
    print(f"Chunked Prefill Profile: {model_name} ({prompt_length} tokens)")
    print(f"{'='*70}")

    tokens = generate_tokens(tokenizer, prompt_length)
    actual_len = len(tokens)

    results = []

    for chunk_size in chunk_sizes:
        times = []

        for run in range(NUM_WARMUP + NUM_RUNS):
            prompt_cache = make_cache(lm_model)
            reset_model_state(lm_model)
            gc.collect()
            mx.clear_cache()

            input_ids = mx.array([tokens])
            processed = 0

            mx.synchronize()
            t0 = time.perf_counter()

            while processed < actual_len:
                chunk = min(chunk_size, actual_len - processed)
                chunk_ids = input_ids[:, processed:processed + chunk]
                outputs = lm_model(chunk_ids, cache=prompt_cache)
                eval_cache_safe(prompt_cache)
                processed += chunk

            t1 = time.perf_counter()
            elapsed = t1 - t0

            if run >= NUM_WARMUP:
                times.append(elapsed)

        avg_time = sum(times) / len(times)
        std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
        tps = actual_len / avg_time

        print(f"  chunk_size={chunk_size:>5}: {avg_time*1000:>8.1f} ms  "
              f"(+/- {std_time*1000:.1f} ms)  "
              f"{tps:>8.1f} t/s")

        results.append({
            "chunk_size": chunk_size,
            "time_ms": avg_time * 1000,
            "std_ms": std_time * 1000,
            "tps": tps,
        })

    return results


def profile_layer_breakdown(lm_model, tokenizer, prompt_length, model_name):
    """Profile individual layer types during prefill."""
    print(f"\n{'='*70}")
    print(f"Layer Breakdown: {model_name} ({prompt_length} tokens)")
    print(f"{'='*70}")

    tokens = generate_tokens(tokenizer, prompt_length)
    input_ids = mx.array([tokens])

    # Get the inner model
    inner_model = lm_model.model if hasattr(lm_model, "model") else lm_model
    layers = inner_model.layers if hasattr(inner_model, "layers") else None

    if layers is None:
        print("  Cannot access individual layers for breakdown.")
        return None

    # Check if this is a hybrid model with linear/full attention layers
    has_linear = any(hasattr(l, "is_linear") for l in layers)

    if not has_linear:
        print("  Pure Transformer model - no linear attention layers.")
        print("  All layers are full attention Transformer layers.")

        # Profile: full model forward pass
        prompt_cache = make_cache(lm_model)
        reset_model_state(lm_model)

        # Warmup
        outputs = lm_model(input_ids, cache=prompt_cache)
        mx.eval(get_logits(outputs))

        prompt_cache = make_cache(lm_model)
        reset_model_state(lm_model)

        mx.synchronize()
        t0 = time.perf_counter()
        outputs = lm_model(input_ids, cache=prompt_cache)
        mx.eval(get_logits(outputs))
        t1 = time.perf_counter()

        print(f"  Full model forward: {(t1-t0)*1000:.1f} ms")
        return {"type": "pure_transformer", "total_ms": (t1-t0)*1000}

    # Hybrid model: separate linear and full attention layers
    linear_layers = [i for i, l in enumerate(layers) if getattr(l, "is_linear", False)]
    full_layers = [i for i, l in enumerate(layers) if not getattr(l, "is_linear", True)]

    print(f"  Linear attention layers: {len(linear_layers)} (indices: {linear_layers[:5]}...)")
    print(f"  Full attention layers: {len(full_layers)} (indices: {full_layers})")

    # Profile layer by layer
    prompt_cache = make_cache(lm_model)
    reset_model_state(lm_model)

    # We need to manually run through the model to time each component
    # First, get embeddings
    if hasattr(inner_model, "embed_tokens"):
        h = inner_model.embed_tokens(input_ids)
    else:
        print("  Cannot access embedding layer.")
        return None

    mx.eval(h)

    # Build masks
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask

    fa_idx = getattr(inner_model, "fa_idx", None)
    ssm_idx = getattr(inner_model, "ssm_idx", None)

    if fa_idx is not None and ssm_idx is not None:
        fa_mask = create_attention_mask(h, prompt_cache[fa_idx])
        ssm_mask = create_ssm_mask(h, prompt_cache[ssm_idx])
    else:
        fa_mask = create_attention_mask(h, None)
        ssm_mask = None

    linear_times = []
    full_times = []

    for i, (layer, c) in enumerate(zip(layers, prompt_cache)):
        is_linear = getattr(layer, "is_linear", False)
        mask = ssm_mask if is_linear else fa_mask

        mx.eval(h)
        mx.synchronize()
        t0 = time.perf_counter()

        h = layer(h, mask, c)
        mx.eval(h)

        t1 = time.perf_counter()
        elapsed = (t1 - t0) * 1000

        if is_linear:
            linear_times.append(elapsed)
        else:
            full_times.append(elapsed)

    avg_linear = sum(linear_times) / len(linear_times) if linear_times else 0
    avg_full = sum(full_times) / len(full_times) if full_times else 0
    total_linear = sum(linear_times)
    total_full = sum(full_times)

    print(f"\n  Linear attention layers ({len(linear_times)}x):")
    print(f"    Per layer: {avg_linear:.2f} ms (range: {min(linear_times):.2f} - {max(linear_times):.2f} ms)")
    print(f"    Total:     {total_linear:.1f} ms")
    for i, t in enumerate(linear_times):
        if i < 3 or i >= len(linear_times) - 1:
            print(f"      Layer {linear_layers[i]:>2}: {t:.2f} ms")
        elif i == 3:
            print(f"      ...")

    print(f"\n  Full attention layers ({len(full_times)}x):")
    print(f"    Per layer: {avg_full:.2f} ms (range: {min(full_times):.2f} - {max(full_times):.2f} ms)")
    print(f"    Total:     {total_full:.1f} ms")
    for i, t in enumerate(full_times):
        print(f"      Layer {full_layers[i]:>2}: {t:.2f} ms")

    print(f"\n  Summary:")
    total = total_linear + total_full
    print(f"    Linear attention: {total_linear:.1f} ms ({100*total_linear/total:.1f}%)")
    print(f"    Full attention:   {total_full:.1f} ms ({100*total_full/total:.1f}%)")
    print(f"    Total layers:     {total:.1f} ms")
    print(f"    Throughput:       {prompt_length/(total/1000):.1f} t/s (layers only)")

    return {
        "type": "hybrid",
        "linear_total_ms": total_linear,
        "full_total_ms": total_full,
        "linear_per_layer_ms": avg_linear,
        "full_per_layer_ms": avg_full,
        "num_linear": len(linear_times),
        "num_full": len(full_times),
        "linear_times": linear_times,
        "full_times": full_times,
    }


def profile_gated_delta_components(lm_model, tokenizer, prompt_lengths, model_name):
    """Profile GatedDeltaNet kernel vs ops for different prompt lengths."""
    print(f"\n{'='*70}")
    print(f"GatedDeltaNet Kernel Profile: {model_name}")
    print(f"{'='*70}")

    inner_model = lm_model.model if hasattr(lm_model, "model") else lm_model
    layers = inner_model.layers if hasattr(inner_model, "layers") else None

    if layers is None:
        print("  Cannot access layers.")
        return None

    # Find first linear layer
    linear_layer = None
    linear_idx = -1
    for i, l in enumerate(layers):
        if getattr(l, "is_linear", False):
            linear_layer = l
            linear_idx = i
            break

    if linear_layer is None:
        print("  No linear attention layers found.")
        return None

    gdn = linear_layer.linear_attn
    print(f"  Using layer {linear_idx} (GatedDeltaNet)")
    print(f"  Config: Hk={gdn.num_k_heads}, Hv={gdn.num_v_heads}, "
          f"Dk={gdn.head_k_dim}, Dv={gdn.head_v_dim}")
    print(f"  Conv kernel: {gdn.conv_kernel_size}")
    print(f"  State shape: [B, {gdn.num_v_heads}, {gdn.head_v_dim}, {gdn.head_k_dim}]")
    state_size_bytes = 1 * gdn.num_v_heads * gdn.head_v_dim * gdn.head_k_dim * 2  # bfloat16
    print(f"  State size per batch: {state_size_bytes / 1024:.1f} KB")

    results = []
    for target_len in prompt_lengths:
        B = 1
        T = target_len
        Hk = gdn.num_k_heads
        Hv = gdn.num_v_heads
        Dk = gdn.head_k_dim
        Dv = gdn.head_v_dim
        hidden = gdn.hidden_size

        # Create synthetic input
        x = mx.random.normal((B, T, hidden))
        x = x.astype(mx.bfloat16) if mx.default_device() == mx.gpu else x

        # Warmup
        norm_out = linear_layer.input_layernorm(x) if hasattr(linear_layer, "input_layernorm") else x
        cache_entry = [None, None]  # [conv_state, recurrent_state]

        mx.eval(norm_out)

        # Time just the GatedDeltaNet forward
        times = []
        for run in range(NUM_WARMUP + NUM_RUNS):
            cache_entry = [None, None]
            mx.eval(norm_out)
            gc.collect()
            mx.clear_cache()

            mx.synchronize()
            t0 = time.perf_counter()
            out = gdn(norm_out, mask=None, cache=cache_entry)
            mx.eval(out)
            t1 = time.perf_counter()

            if run >= NUM_WARMUP:
                times.append(t1 - t0)

        avg = sum(times) / len(times)
        std = (sum((t - avg) ** 2 for t in times) / len(times)) ** 0.5
        tps = T / avg

        print(f"  T={T:>5}: {avg*1000:>8.2f} ms (+/- {std*1000:.2f} ms)  {tps:>8.1f} t/s")
        results.append({"length": T, "time_ms": avg * 1000, "std_ms": std * 1000, "tps": tps})

    return results


def profile_scaling_analysis(results_list, model_name):
    """Analyze whether prefill scales linearly or super-linearly with prompt length."""
    print(f"\n{'='*70}")
    print(f"Scaling Analysis: {model_name}")
    print(f"{'='*70}")

    if len(results_list) < 2:
        print("  Not enough data points for scaling analysis.")
        return

    # Analyze time/token ratio
    print(f"\n  {'Length':>8}  {'Time (ms)':>10}  {'t/s':>8}  {'ms/tok':>8}  {'Ratio vs 64':>12}")
    base_ms_per_tok = None
    for r in results_list:
        ms_per_tok = r["time_ms"] / r["length"]
        if base_ms_per_tok is None:
            base_ms_per_tok = ms_per_tok
            ratio = 1.0
        else:
            ratio = ms_per_tok / base_ms_per_tok

        print(f"  {r['length']:>8}  {r['time_ms']:>10.1f}  {r['tps']:>8.1f}  "
              f"{ms_per_tok:>8.3f}  {ratio:>12.2f}x")

    # Check if scaling is quadratic
    if len(results_list) >= 3:
        lengths = [r["length"] for r in results_list]
        times = [r["time_ms"] for r in results_list]

        # Fit: time = a * length + b * length^2
        # If b >> a, scaling is quadratic (attention-dominated)
        # If b << a, scaling is linear (GatedDeltaNet-dominated or compute-bound)
        n = len(lengths)
        sum_l = sum(lengths)
        sum_l2 = sum(l**2 for l in lengths)
        sum_l3 = sum(l**3 for l in lengths)
        sum_l4 = sum(l**4 for l in lengths)
        sum_tl = sum(t*l for t, l in zip(times, lengths))
        sum_tl2 = sum(t*l**2 for t, l in zip(times, lengths))

        # Solve 2x2 system: [sum_l2, sum_l3; sum_l3, sum_l4] * [a, b] = [sum_tl, sum_tl2]
        det = sum_l2 * sum_l4 - sum_l3 * sum_l3
        if abs(det) > 1e-10:
            a = (sum_tl * sum_l4 - sum_tl2 * sum_l3) / det
            b = (sum_l2 * sum_tl2 - sum_l3 * sum_tl) / det

            print(f"\n  Fit: time_ms = {a:.4f} * L + {b:.8f} * L^2")
            if abs(a) > 1e-10:
                crossover = -a / (2 * b) if b != 0 else float("inf")
                print(f"  Linear coefficient: {a:.4f} ms/token")
                print(f"  Quadratic coefficient: {b:.8f} ms/token^2")
                if b > 0:
                    print(f"  Quadratic term dominates at L > {a/b:.0f} tokens")
                else:
                    print(f"  Scaling is sub-quadratic (good!)")


def main():
    print("=" * 70)
    print("MLX Prefill Performance Profiler")
    print("=" * 70)

    # ---- Profile Qwen3.5 (Hybrid Architecture) ----
    print("\n\nLoading Qwen3.5-9B-8bit (hybrid: 75% GatedDeltaNet + 25% Transformer)...")
    try:
        model_35, tokenizer_35, loader_35 = load_model_and_tokenizer(QWEN35_PATH)
        lm_35 = get_language_model(model_35, loader_35)
        print(f"  Loaded. Peak memory: {mx.get_peak_memory() / 1e9:.2f} GB")

        # 1. Basic prefill profile
        results_35 = profile_prefill(lm_35, tokenizer_35, PROMPT_LENGTHS, "Qwen3.5-9B-8bit", loader_35)

        # 2. Layer breakdown at 512 tokens
        breakdown_35 = profile_layer_breakdown(lm_35, tokenizer_35, 512, "Qwen3.5-9B-8bit")

        # 3. GatedDeltaNet component profile
        gdn_results = profile_gated_delta_components(lm_35, tokenizer_35, PROMPT_LENGTHS, "Qwen3.5-9B-8bit")

        # 4. Scaling analysis
        profile_scaling_analysis(results_35, "Qwen3.5-9B-8bit")

        # 5. Chunked prefill comparison
        chunked_35 = profile_chunked_prefill(lm_35, tokenizer_35, 1024,
                                             [64, 128, 256, 512, 1024], "Qwen3.5-9B-8bit")

        # Free memory
        del model_35, lm_35, tokenizer_35
        gc.collect()
        mx.clear_cache()

    except Exception as e:
        print(f"  ERROR loading Qwen3.5: {e}")
        import traceback
        traceback.print_exc()
        results_35 = None

    # ---- Profile Qwen3 (Pure Transformer) ----
    print("\n\nLoading Qwen3-8B-4bit (pure Transformer)...")
    try:
        model_3, tokenizer_3, loader_3 = load_model_and_tokenizer(QWEN3_PATH)
        lm_3 = get_language_model(model_3, loader_3)
        print(f"  Loaded. Peak memory: {mx.get_peak_memory() / 1e9:.2f} GB")

        # 1. Basic prefill profile
        results_3 = profile_prefill(lm_3, tokenizer_3, PROMPT_LENGTHS, "Qwen3-8B-4bit", loader_3)

        # 2. Scaling analysis
        profile_scaling_analysis(results_3, "Qwen3-8B-4bit")

        # Free memory
        del model_3, lm_3, tokenizer_3
        gc.collect()
        mx.clear_cache()

    except Exception as e:
        print(f"  ERROR loading Qwen3: {e}")
        import traceback
        traceback.print_exc()
        results_3 = None

    # ---- Comparison ----
    if results_35 and results_3:
        print(f"\n{'='*70}")
        print("Comparison: Qwen3.5 vs Qwen3 Prefill")
        print(f"{'='*70}")

        print(f"\n  {'Length':>8}  {'Qwen3.5 t/s':>12}  {'Qwen3 t/s':>10}  {'Ratio':>8}")
        for r35, r3 in zip(results_35, results_3):
            if r35["length"] == r3["length"]:
                ratio = r3["tps"] / r35["tps"] if r35["tps"] > 0 else 0
                print(f"  {r35['length']:>8}  {r35['tps']:>12.1f}  {r3['tps']:>10.1f}  {ratio:>7.2f}x")

    print("\n\nDone.")


if __name__ == "__main__":
    main()
