"""
Profile single-token vs batch forward pass time.
Measures the actual GPU time for different batch sizes to determine
if batch verification in speculative decoding is worthwhile.
"""
import sys
import time
sys.path.insert(0, "/Users/alex/Documents/Codes/RefSources/mlx-vlm-optimized")

import mlx.core as mx
from mlx_vlm import load
from mlx_vlm.models import cache
from mlx_vlm.generate import generation_stream

from cache_utils import save_cache_checkpoint, restore_cache_checkpoint

MODEL_PATH = "/Users/alex/Documents/mlx-community/Qwen3.5-9B-8bit"

def main():
    print("Loading model...")
    model, processor = load(MODEL_PATH)
    print(f"Model loaded. Peak memory: {mx.get_peak_memory() / 1e9:.2f} GB\n")

    # Prepare a prompt
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    prompt = "Explain the concept of attention mechanisms in neural networks."
    input_ids = mx.array([tokenizer.encode(prompt)])

    prompt_cache = cache.make_prompt_cache(model.language_model)

    # Prefill
    with mx.stream(generation_stream):
        embedding_output = model.get_input_embeddings(input_ids, None, mask=None)
        inputs_embeds = embedding_output.inputs_embeds
        outputs = model.language_model(
            input_ids,
            inputs_embeds=inputs_embeds,
            cache=prompt_cache,
        )
        first_token = mx.argmax(outputs.logits[:, -1, :], axis=-1)
    mx.eval(first_token)

    # Generate a few tokens to warm up
    y = first_token
    for _ in range(20):
        with mx.stream(generation_stream):
            outputs = model.language_model(y[None], cache=prompt_cache)
            y = mx.argmax(outputs.logits[:, -1, :], axis=-1)
        mx.eval(y)

    print("Warmup complete. Profiling forward passes...\n")

    # === Profile: Single token forward pass ===
    times_single = []
    for _ in range(20):
        token = mx.array([[42]])  # arbitrary token

        # Checkpoint before
        cp = save_cache_checkpoint(prompt_cache)

        mx.synchronize()
        t0 = time.perf_counter()
        with mx.stream(generation_stream):
            outputs = model.language_model(token, cache=prompt_cache)
            logits = outputs.logits[:, -1, :]
        mx.eval(logits)
        t1 = time.perf_counter()
        times_single.append((t1 - t0) * 1000)

        # Restore cache
        restore_cache_checkpoint(prompt_cache, cp)

    avg_single = sum(times_single[5:]) / len(times_single[5:])
    print(f"Single token forward:  {avg_single:.2f} ms  (stddev: {(sum((t-avg_single)**2 for t in times_single[5:])/len(times_single[5:]))**0.5:.2f})")

    # === Profile: Batch forward passes (2, 3, 5, 8 tokens) ===
    for batch_size in [2, 3, 5, 8]:
        times_batch = []
        for _ in range(20):
            tokens = mx.array([[42] * batch_size])

            cp = save_cache_checkpoint(prompt_cache)

            mx.synchronize()
            t0 = time.perf_counter()
            with mx.stream(generation_stream):
                outputs = model.language_model(tokens, cache=prompt_cache)
                all_logits = outputs.logits  # [1, batch_size, vocab]
            mx.eval(all_logits)
            t1 = time.perf_counter()
            times_batch.append((t1 - t0) * 1000)

            restore_cache_checkpoint(prompt_cache, cp)

        avg_batch = sum(times_batch[5:]) / len(times_batch[5:])
        speedup = batch_size * avg_single / avg_batch
        print(f"Batch({batch_size}) forward:       {avg_batch:.2f} ms  "
              f"(speedup vs {batch_size}×single: {speedup:.2f}x, "
              f"per-token: {avg_batch/batch_size:.2f} ms)")

    # === Profile: Checkpoint + Batch + Reprocess ===
    print("\n--- Full speculative step (checkpoint + batch verify + reprocess) ---")
    for batch_size in [3, 5]:
        times_full = []
        for _ in range(20):
            t0_total = time.perf_counter()

            # 1. Checkpoint
            mx.synchronize()
            t_cp_start = time.perf_counter()
            cp = save_cache_checkpoint(prompt_cache)
            t_cp_end = time.perf_counter()

            # 2. Batch forward
            tokens = mx.array([[42] * batch_size])
            t_fwd_start = time.perf_counter()
            with mx.stream(generation_stream):
                outputs = model.language_model(tokens, cache=prompt_cache)
                all_logits = outputs.logits
            mx.eval(all_logits)
            t_fwd_end = time.perf_counter()

            # 3. Sampling (batch)
            t_sample_start = time.perf_counter()
            sampled = mx.argmax(all_logits[0], axis=-1)
            mx.eval(sampled)
            t_sample_end = time.perf_counter()

            # 4. Restore
            t_restore_start = time.perf_counter()
            restore_cache_checkpoint(prompt_cache, cp)
            t_restore_end = time.perf_counter()

            # 5. Reprocess (half the batch = partial acceptance)
            n_accept = batch_size // 2
            reprocess = mx.array([[42] * (n_accept + 1)])
            t_repr_start = time.perf_counter()
            with mx.stream(generation_stream):
                model.language_model(reprocess, cache=prompt_cache)
                mx.eval([c.state for c in prompt_cache])
            t_repr_end = time.perf_counter()

            # Restore again for next iteration
            restore_cache_checkpoint(prompt_cache, cp)

            t1_total = time.perf_counter()

            times_full.append({
                'total': (t1_total - t0_total) * 1000,
                'checkpoint': (t_cp_end - t_cp_start) * 1000,
                'batch_fwd': (t_fwd_end - t_fwd_start) * 1000,
                'sampling': (t_sample_end - t_sample_start) * 1000,
                'restore': (t_restore_end - t_restore_start) * 1000,
                'reprocess': (t_repr_end - t_repr_start) * 1000,
            })

        valid = times_full[5:]
        print(f"\nBatch({batch_size}), accept {batch_size//2}:")
        for key in ['checkpoint', 'batch_fwd', 'sampling', 'restore', 'reprocess', 'total']:
            avg = sum(t[key] for t in valid) / len(valid)
            print(f"  {key:>12}: {avg:6.2f} ms")

        equiv_tokens = batch_size // 2 + 1  # accepted + bonus
        total_avg = sum(t['total'] for t in valid) / len(valid)
        print(f"  → {equiv_tokens} tokens in {total_avg:.1f} ms = {total_avg/equiv_tokens:.1f} ms/tok "
              f"(std: {avg_single:.1f} ms/tok)")

if __name__ == "__main__":
    main()
