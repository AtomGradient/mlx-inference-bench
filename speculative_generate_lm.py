"""
Prompt Lookup Speculative Decoding for mlx-lm (pure Transformer models).

Adapted from speculative_generate.py (mlx-vlm version) for use with mlx-lm
models like Qwen3 that are pure Transformer architectures with KVCache.

Key differences from VLM version:
- Uses model directly (not model.language_model)
- No pixel_values/mask/vision processing
- Model returns logits directly (not .logits attribute)
- KVCache supports trim, so we can use trim instead of checkpoint/restore
  (but we still use checkpoint/restore for code consistency and correctness)

Usage:
    from speculative_generate_lm import speculative_generate_lm
    result = speculative_generate_lm(model, tokenizer, prompt, ...)
"""

import contextlib
import functools
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_reduce
from mlx_lm.models import cache
from mlx_lm.generate import (
    generation_stream,
    maybe_quantize_kv_cache,
    wired_limit,
)
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper

from cache_utils import save_cache_checkpoint, restore_cache_checkpoint


# --- N-gram Lookup (identical to VLM version) ---


class NgramLookup:
    """
    Fast N-gram lookup table for prompt lookup decoding.

    Builds a hash map from N-grams to their continuations in the source tokens.
    """

    def __init__(self, ngram_size: int = 3, max_draft: int = 5):
        self.ngram_size = ngram_size
        self.max_draft = max_draft
        self._table: Dict[tuple, List[int]] = defaultdict(list)

    def build(self, tokens: List[int]) -> None:
        """Build N-gram table from a list of tokens."""
        self._table.clear()
        n = self.ngram_size
        for i in range(len(tokens) - n):
            key = tuple(tokens[i : i + n])
            self._table[key].append(i + n)

    def update(self, tokens: List[int], start_pos: int) -> None:
        """Incrementally update the N-gram table with new tokens."""
        n = self.ngram_size
        begin = max(0, start_pos - n + 1)
        for i in range(begin, len(tokens) - n):
            key = tuple(tokens[i : i + n])
            pos = i + n
            if pos not in self._table.get(key, []):
                self._table[key].append(pos)

    def lookup(self, context: List[int], all_tokens: List[int]) -> List[int]:
        """
        Find draft tokens by matching the last N tokens of context.

        Returns up to max_draft continuation tokens, or empty list if no match.
        """
        n = self.ngram_size
        if len(context) < n:
            return []

        key = tuple(context[-n:])
        positions = self._table.get(key, [])

        if not positions:
            return []

        best_draft = []
        for pos in positions:
            draft = []
            for j in range(self.max_draft):
                if pos + j < len(all_tokens):
                    draft.append(all_tokens[pos + j])
                else:
                    break
            if len(draft) > len(best_draft):
                best_draft = draft

        return best_draft


# --- Speculative Stats ---


@dataclass
class SpeculativeStats:
    """Statistics for speculative decoding."""

    total_tokens: int = 0
    draft_tokens_proposed: int = 0
    draft_tokens_accepted: int = 0
    num_speculative_steps: int = 0
    num_normal_steps: int = 0

    @property
    def acceptance_rate(self) -> float:
        if self.draft_tokens_proposed == 0:
            return 0.0
        return self.draft_tokens_accepted / self.draft_tokens_proposed

    @property
    def tokens_per_step(self) -> float:
        total_steps = self.num_speculative_steps + self.num_normal_steps
        if total_steps == 0:
            return 0.0
        return self.total_tokens / total_steps


# --- Speculative Generation Step (adapted for mlx-lm) ---


def speculative_generate_step_lm(
    prompt: mx.array,
    model: nn.Module,
    *,
    max_tokens: int = 256,
    sampler: Optional[Callable[[mx.array], mx.array]] = None,
    logits_processors: Optional[
        List[Callable[[mx.array, mx.array], mx.array]]
    ] = None,
    max_kv_size: Optional[int] = None,
    prompt_cache: Optional[List[Any]] = None,
    prefill_step_size: int = 2048,
    kv_bits: Optional[int] = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = 0,
    # Speculative decoding parameters
    ngram_size: int = 3,
    max_draft_tokens: int = 5,
) -> Generator[Tuple[int, mx.array, bool, Optional[SpeculativeStats]], None, None]:
    """
    A generator producing tokens using prompt lookup speculative decoding
    for pure Transformer (mlx-lm) models.

    Key difference from VLM version: model is called directly and returns
    logits tensor (not an object with .logits attribute).

    Args:
        prompt: The input prompt token ids (1D array).
        model: The language model (mlx-lm Model).
        max_tokens: Maximum tokens to generate.
        sampler: Token sampler function.
        ngram_size: Size of N-grams for lookup (default: 3).
        max_draft_tokens: Maximum draft tokens per step (default: 5).

    Yields:
        Tuple of (token_id, logprobs, from_draft, stats).
    """
    quantize_cache_fn = functools.partial(
        maybe_quantize_kv_cache,
        quantized_kv_start=quantized_kv_start,
        kv_group_size=kv_group_size,
        kv_bits=kv_bits,
    )

    if sampler is None:
        sampler = make_sampler(0.0)

    tokens = None

    # Create the KV cache for generation
    if prompt_cache is None:
        prompt_cache = cache.make_prompt_cache(
            model,
            max_kv_size=max_kv_size,
        )

    # Build N-gram lookup from prompt tokens
    prompt_tokens = prompt.tolist()
    ngram = NgramLookup(ngram_size=ngram_size, max_draft=max_draft_tokens)
    ngram.build(prompt_tokens)

    stats = SpeculativeStats()
    generated_tokens: List[int] = []
    all_tokens = list(prompt_tokens)  # prompt + generated

    def _process_logits(y_input, logits):
        """Apply processors and sample from logits."""
        nonlocal tokens
        if logits_processors and y_input is not None and len(y_input) > 0:
            tokens = (
                mx.concat([tokens, y_input.flatten()])
                if tokens is not None
                else y_input.flatten()
            )
            for processor in logits_processors:
                logits = processor(tokens, logits)
        logprobs = logits - mx.logsumexp(logits, keepdims=True)
        sampled = sampler(logprobs)
        return sampled, logprobs

    def _step_single(y_token):
        """Single-token generation step."""
        with mx.stream(generation_stream):
            # mlx-lm model: model(input_ids, cache=cache) -> logits
            logits = model(y_token, cache=prompt_cache)
            logits = logits[:, -1, :]
            quantize_cache_fn(prompt_cache)
            sampled, logprobs = _process_logits(y_token, logits)
            return sampled, logprobs.squeeze(0)

    def _step_multi(y_tokens):
        """
        Multi-token forward pass for speculative verification.
        Returns logits for ALL positions (not just last).
        """
        with mx.stream(generation_stream):
            all_logits = model(y_tokens, cache=prompt_cache)
            quantize_cache_fn(prompt_cache)
            return all_logits

    # === Prefill Phase ===
    y = prompt
    with mx.stream(generation_stream):
        # Chunked prefill for large prompts
        while len(y) > prefill_step_size:
            model(y[:prefill_step_size][None], cache=prompt_cache)
            quantize_cache_fn(prompt_cache)
            mx.eval([c.state for c in prompt_cache])
            y = y[prefill_step_size:]
            mx.clear_cache()

        # Process last chunk and get first token
        logits = model(y[None], cache=prompt_cache)
        logits = logits[:, -1, :]
        quantize_cache_fn(prompt_cache)
        logprobs = logits - mx.logsumexp(logits, keepdims=True)
        first_y = sampler(logprobs)

    mx.async_eval(first_y)
    y = first_y

    # === Generation Phase with Speculative Decoding ===
    n = 0
    while True:
        # 1. Start GPU computation for next token FIRST (standard pipeline)
        if n != max_tokens:
            next_y, next_logprobs = _step_single(y[None])
            mx.async_eval(next_y)

        # 2. Wait for first token
        if n == 0:
            mx.eval(y)
        if n == max_tokens:
            break

        # 3. Get current token value
        token_id = y.item()
        generated_tokens.append(token_id)
        all_tokens.append(token_id)
        stats.total_tokens += 1

        # 4. Yield current token (GPU computing next_y in background)
        yield token_id, logprobs, False, stats
        n += 1

        if n >= max_tokens:
            break

        # 5. After yield: N-gram lookup (GPU still working on next_y)
        ngram.update(all_tokens, len(all_tokens) - 1)
        draft_tokens = ngram.lookup(all_tokens, all_tokens)
        if draft_tokens:
            draft_tokens = draft_tokens[: max_tokens - n]

        if not draft_tokens:
            # --- Standard path: use already-computed next_y ---
            y, logprobs = next_y, next_logprobs
            stats.num_normal_steps += 1
        else:
            # --- Speculative path ---
            mx.eval(next_y)
            next_tid = next_y.item()

            if next_tid != draft_tokens[0]:
                # No match: standard path
                y, logprobs = next_y, next_logprobs
                stats.num_normal_steps += 1
                stats.num_speculative_steps += 1
                stats.draft_tokens_proposed += 1
            else:
                # First draft matches! Accept it and try remaining.
                stats.num_speculative_steps += 1
                stats.draft_tokens_proposed += len(draft_tokens)
                stats.draft_tokens_accepted += 1

                # Yield the matched token (next_y)
                generated_tokens.append(next_tid)
                all_tokens.append(next_tid)
                stats.total_tokens += 1
                yield next_tid, next_logprobs, True, stats
                n += 1

                if n >= max_tokens or len(draft_tokens) <= 1:
                    if n >= max_tokens:
                        break
                    y, logprobs = _step_single(next_y.reshape(1, -1))
                    mx.eval(y)
                    continue

                # Verify remaining drafts in batch
                remaining = draft_tokens[1: max_tokens - n + 1]
                if not remaining:
                    y, logprobs = _step_single(next_y.reshape(1, -1))
                    mx.eval(y)
                    continue

                # Save cache checkpoint (for KVCache: save offset)
                checkpoint = save_cache_checkpoint(prompt_cache)

                # Batch forward: [next_tid, remaining_0, ..., remaining_N]
                verify_input = mx.array(
                    [[next_tid] + list(remaining)], dtype=mx.uint32
                )
                all_logits = _step_multi(verify_input)

                # Batch sampling: compare all positions at once
                nr = len(remaining)
                draft_logits = all_logits[0, :nr, :]
                draft_logprobs = draft_logits - mx.logsumexp(
                    draft_logits, axis=-1, keepdims=True
                )
                draft_sampled = sampler(draft_logprobs)
                draft_expected = mx.array(remaining)
                matches = draft_sampled.flatten() == draft_expected
                mx.eval(matches, draft_sampled)

                # Find first mismatch
                matches_list = matches.tolist()
                n_accepted = 0
                for m in matches_list:
                    if m:
                        n_accepted += 1
                    else:
                        break

                # Bonus token from position after last accepted
                bonus_logits = all_logits[0:1, n_accepted, :]
                bonus_lp = bonus_logits - mx.logsumexp(bonus_logits)
                bonus_y = sampler(bonus_lp)
                mx.eval(bonus_y)
                stats.draft_tokens_accepted += n_accepted

                # Fix cache: restore + reprocess accepted prefix
                if n_accepted < len(remaining):
                    restore_cache_checkpoint(prompt_cache, checkpoint)
                    reprocess = mx.array(
                        [[next_tid] + list(remaining[:n_accepted])],
                        dtype=mx.uint32,
                    )
                    with mx.stream(generation_stream):
                        model(reprocess, cache=prompt_cache)
                        quantize_cache_fn(prompt_cache)
                        mx.eval([c.state for c in prompt_cache])

                # Yield accepted remaining drafts
                for i in range(n_accepted):
                    if n >= max_tokens:
                        break
                    dtid = remaining[i]
                    generated_tokens.append(dtid)
                    all_tokens.append(dtid)
                    stats.total_tokens += 1
                    yield dtid, draft_logprobs[i], True, stats
                    n += 1

                if n >= max_tokens:
                    break

                y, logprobs = bonus_y, bonus_lp.squeeze(0)

        if n % 256 == 0:
            mx.clear_cache()


# --- High-level API ---


@dataclass
class SpeculativeGenerationResult:
    """Result from speculative generation."""

    text: str = ""
    token: Optional[int] = None
    logprobs: Optional[mx.array] = None
    from_draft: bool = False
    prompt_tokens: int = 0
    generation_tokens: int = 0
    total_tokens: int = 0
    prompt_tps: float = 0.0
    generation_tps: float = 0.0
    peak_memory: float = 0.0
    acceptance_rate: float = 0.0
    tokens_per_step: float = 0.0


def speculative_stream_generate_lm(
    model: nn.Module,
    tokenizer: Union[TokenizerWrapper, Any],
    prompt: Union[str, mx.array, List[int]],
    *,
    max_tokens: int = 256,
    ngram_size: int = 3,
    max_draft_tokens: int = 5,
    **kwargs,
) -> Generator[SpeculativeGenerationResult, None, None]:
    """
    A generator producing text using prompt lookup speculative decoding
    for mlx-lm models.

    Args:
        model: The language model.
        tokenizer: The tokenizer (TokenizerWrapper).
        prompt: The input prompt (string, array, or token list).
        max_tokens: Maximum tokens to generate.
        ngram_size: Size of N-grams for lookup (default: 3).
        max_draft_tokens: Maximum draft tokens per step (default: 5).

    Yields:
        SpeculativeGenerationResult with text, statistics, and metadata.
    """
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    if not isinstance(prompt, mx.array):
        if isinstance(prompt, str):
            add_special_tokens = tokenizer.bos_token is None or not prompt.startswith(
                tokenizer.bos_token
            )
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        else:
            prompt_tokens = list(prompt)
        input_ids = mx.array(prompt_tokens)
    else:
        input_ids = prompt

    detokenizer = tokenizer.detokenizer

    with wired_limit(model, [generation_stream]):
        detokenizer.reset()

        gen = speculative_generate_step_lm(
            input_ids,
            model,
            max_tokens=max_tokens,
            ngram_size=ngram_size,
            max_draft_tokens=max_draft_tokens,
            **kwargs,
        )
        tic = time.perf_counter()

        for n, (token, logprobs, from_draft, spec_stats) in enumerate(gen):
            if n == 0:
                prompt_time = time.perf_counter() - tic
                prompt_tps = input_ids.size / prompt_time
                tic = time.perf_counter()

            if token in tokenizer.eos_token_ids:
                break

            detokenizer.add_token(token)

            gen_time = time.perf_counter() - tic
            yield SpeculativeGenerationResult(
                text=detokenizer.last_segment,
                token=token,
                logprobs=logprobs,
                from_draft=from_draft,
                prompt_tokens=input_ids.size,
                generation_tokens=n + 1,
                total_tokens=input_ids.size + n + 1,
                prompt_tps=prompt_tps,
                generation_tps=(n + 1) / gen_time if gen_time > 0 else 0,
                peak_memory=mx.get_peak_memory() / 1e9,
                acceptance_rate=spec_stats.acceptance_rate,
                tokens_per_step=spec_stats.tokens_per_step,
            )

        detokenizer.finalize()
        gen_time = time.perf_counter() - tic
        yield SpeculativeGenerationResult(
            text=detokenizer.last_segment,
            token=token,
            logprobs=logprobs,
            from_draft=from_draft,
            prompt_tokens=input_ids.size,
            generation_tokens=n + 1,
            total_tokens=input_ids.size + n + 1,
            prompt_tps=prompt_tps,
            generation_tps=(n + 1) / gen_time if gen_time > 0 else 0,
            peak_memory=mx.get_peak_memory() / 1e9,
            acceptance_rate=spec_stats.acceptance_rate,
            tokens_per_step=spec_stats.tokens_per_step,
        )

        mx.clear_cache()


def speculative_generate_lm(
    model: nn.Module,
    tokenizer: Union[TokenizerWrapper, Any],
    prompt: Union[str, mx.array, List[int]],
    verbose: bool = False,
    *,
    max_tokens: int = 256,
    ngram_size: int = 3,
    max_draft_tokens: int = 5,
    **kwargs,
) -> SpeculativeGenerationResult:
    """
    Generate text using prompt lookup speculative decoding for mlx-lm models.

    Args:
        model: The language model.
        tokenizer: The tokenizer.
        prompt: The input prompt.
        verbose: Print tokens and timing info.
        max_tokens: Maximum tokens to generate.
        ngram_size: N-gram size for draft lookup.
        max_draft_tokens: Max draft tokens per speculative step.

    Returns:
        SpeculativeGenerationResult with full generation statistics.
    """
    if verbose:
        print("=" * 10)
        print("Prompt:", prompt[:200] if isinstance(prompt, str) else f"[{len(prompt)} tokens]")
        print(f"[Speculative] ngram_size={ngram_size}, max_draft={max_draft_tokens}")

    text = ""
    last_response = None

    for response in speculative_stream_generate_lm(
        model,
        tokenizer,
        prompt,
        max_tokens=max_tokens,
        ngram_size=ngram_size,
        max_draft_tokens=max_draft_tokens,
        **kwargs,
    ):
        if verbose:
            print(response.text, end="", flush=True)
        text += response.text
        last_response = response

    if verbose:
        print("\n" + "=" * 10)
        if len(text) == 0 or last_response is None:
            print("No text generated for this prompt")
            return SpeculativeGenerationResult(
                text=text,
                peak_memory=mx.get_peak_memory() / 1e9,
            )
        print(
            f"Prompt: {last_response.prompt_tokens} tokens, "
            f"{last_response.prompt_tps:.3f} tokens-per-sec"
        )
        print(
            f"Generation: {last_response.generation_tokens} tokens, "
            f"{last_response.generation_tps:.3f} tokens-per-sec"
        )
        print(f"Peak memory: {last_response.peak_memory:.3f} GB")
        print(
            f"Acceptance rate: {last_response.acceptance_rate:.1%}, "
            f"Tokens/step: {last_response.tokens_per_step:.2f}"
        )

    if last_response is None:
        return SpeculativeGenerationResult(text=text, peak_memory=mx.get_peak_memory() / 1e9)

    return SpeculativeGenerationResult(
        text=text,
        token=last_response.token,
        logprobs=last_response.logprobs,
        from_draft=last_response.from_draft,
        prompt_tokens=last_response.prompt_tokens,
        generation_tokens=last_response.generation_tokens,
        total_tokens=last_response.total_tokens,
        prompt_tps=last_response.prompt_tps,
        generation_tps=last_response.generation_tps,
        peak_memory=last_response.peak_memory,
        acceptance_rate=last_response.acceptance_rate,
        tokens_per_step=last_response.tokens_per_step,
    )
