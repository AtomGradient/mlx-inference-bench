"""
Prompt Lookup Speculative Decoding for mlx-vlm (Qwen3.5 optimized).

This module implements N-gram based speculative decoding without a draft model.
It uses patterns from the prompt and previously generated tokens to predict
upcoming tokens, then verifies them in a single batched forward pass.

Key insight for Qwen3.5: The GatedDeltaNet recurrence is <0.2% of per-token
compute. The main cost is matmul projections and MLP, which benefit greatly
from batched processing. So speculative decoding is effective even with
non-trimmable ArraysCache - we use checkpoint/restore instead.

Usage:
    from speculative_generate import speculative_stream_generate
    for result in speculative_stream_generate(model, processor, prompt, ...):
        print(result.text, end="", flush=True)
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
from mlx_lm.generate import maybe_quantize_kv_cache
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from mlx_vlm.models import cache
from mlx_vlm.generate import (
    DEFAULT_PREFILL_STEP_SIZE,
    DEFAULT_QUANTIZED_KV_START,
    GenerationResult,
    generation_stream,
    wired_limit,
)
from mlx_vlm.utils import (
    StoppingCriteria,
    ThinkingBudgetCriteria,
    prepare_inputs,
)

from cache_utils import (
    save_cache_checkpoint,
    restore_cache_checkpoint,
)



# --- N-gram Lookup ---


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
            # Store the position after the N-gram
            self._table[key].append(i + n)

    def update(self, tokens: List[int], start_pos: int) -> None:
        """Incrementally update the N-gram table with new tokens."""
        n = self.ngram_size
        # Add N-grams that end at or after start_pos
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

        # Use the longest matching continuation from any position
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


# --- Speculative Generation ---


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


def speculative_generate_step(
    input_ids: mx.array,
    model: nn.Module,
    pixel_values,
    mask,
    *,
    max_tokens: int = 256,
    temperature: float = 0.0,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: Optional[int] = 20,
    top_p: float = 1.0,
    logit_bias: Optional[Dict[int, float]] = None,
    prompt_cache: Optional[List[Any]] = None,
    max_kv_size: Optional[int] = None,
    kv_bits: Optional[int] = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = 0,
    sampler: Optional[Callable[[mx.array], mx.array]] = None,
    logits_processors: Optional[
        List[Callable[[mx.array, mx.array], mx.array]]
    ] = None,
    prefill_step_size: Optional[int] = DEFAULT_PREFILL_STEP_SIZE,
    # Speculative decoding parameters
    ngram_size: int = 3,
    max_draft_tokens: int = 5,
    **kwargs,
) -> Generator[Tuple[int, mx.array, bool, Optional[SpeculativeStats]], None, None]:
    """
    A generator producing tokens using prompt lookup speculative decoding.

    Uses N-gram matching from the prompt and generated text to draft tokens,
    then verifies them in a single batched forward pass.

    Args:
        input_ids: The input prompt token ids.
        model: The VLM model.
        pixel_values: Vision inputs.
        mask: Attention mask.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.
        ngram_size: Size of N-grams for lookup (default: 3).
        max_draft_tokens: Maximum draft tokens per step (default: 5).
        **kwargs: Additional arguments passed to model.

    Yields:
        Tuple of (token_id, logprobs, from_draft, stats).
        from_draft is True if the token was a verified draft token.
        stats is the cumulative SpeculativeStats (on last yield per step).
    """
    quantize_cache_fn = functools.partial(
        maybe_quantize_kv_cache,
        quantized_kv_start=quantized_kv_start,
        kv_group_size=kv_group_size,
        kv_bits=kv_bits,
    )

    if sampler is None:
        sampler = make_sampler(temperature, top_p)

    processors = make_logits_processors(
        logit_bias, repetition_penalty, repetition_context_size
    )
    if logits_processors is not None:
        processors.extend(logits_processors)

    y = input_ids
    tokens = mx.array([], dtype=input_ids.dtype)

    thinking_budget_criteria = kwargs.pop("thinking_budget_criteria", None)

    # Create the KV cache for generation
    if prompt_cache is None:
        prompt_cache = cache.make_prompt_cache(
            model.language_model,
            max_kv_size=max_kv_size,
        )

    # Build N-gram lookup from prompt tokens
    prompt_tokens = input_ids.flatten().tolist()
    ngram = NgramLookup(ngram_size=ngram_size, max_draft=max_draft_tokens)
    ngram.build(prompt_tokens)

    stats = SpeculativeStats()
    generated_tokens: List[int] = []
    all_tokens = list(prompt_tokens)  # prompt + generated

    def _process_logits(y_input, logits):
        """Apply processors and sample from logits."""
        nonlocal tokens
        if len(processors) > 0 and y_input is not None and len(y_input) > 0:
            tokens = mx.concat([tokens, y_input.flatten()])
            for processor in processors:
                logits = processor(tokens, logits)
        logprobs = logits - mx.logsumexp(logits)
        sampled = sampler(logprobs)
        return sampled, logprobs

    def _step_single(y_token):
        """Single-token generation step (standard path)."""
        nonlocal kwargs
        with mx.stream(generation_stream):
            if "decoder_input_ids" in kwargs:
                outputs = model.language_model(
                    cache=prompt_cache,
                    **kwargs,
                )
            else:
                outputs = model.language_model(
                    y_token,
                    cache=prompt_cache,
                    **kwargs,
                )
            logits = outputs.logits[:, -1, :]
            quantize_cache_fn(prompt_cache)

            sampled, logprobs = _process_logits(y_token, logits)

            if outputs.cross_attention_states is not None:
                kwargs = {"cross_attention_states": outputs.cross_attention_states}
            elif outputs.encoder_outputs is not None:
                kwargs = {"encoder_outputs": outputs.encoder_outputs}
            else:
                kwargs = {}

            return sampled, logprobs.squeeze(0)

    def _step_multi(y_tokens):
        """
        Multi-token forward pass for speculative verification.
        Returns logits for ALL positions (not just last).
        """
        nonlocal kwargs
        with mx.stream(generation_stream):
            outputs = model.language_model(
                y_tokens,
                cache=prompt_cache,
                **kwargs,
            )
            all_logits = outputs.logits  # [B, seq_len, vocab_size]
            quantize_cache_fn(prompt_cache)

            if outputs.cross_attention_states is not None:
                kwargs = {"cross_attention_states": outputs.cross_attention_states}
            elif outputs.encoder_outputs is not None:
                kwargs = {"encoder_outputs": outputs.encoder_outputs}
            else:
                kwargs = {}

            return all_logits

    # === Prefill Phase (same as generate_step) ===
    with mx.stream(generation_stream):
        embedding_output = model.get_input_embeddings(
            input_ids, pixel_values, mask=mask, **kwargs
        )
        inputs_embeds = embedding_output.inputs_embeds
        kwargs.update(
            {
                k: v
                for k, v in embedding_output.to_dict().items()
                if k != "inputs_embeds" and v is not None
            }
        )

        if (
            prefill_step_size is not None
            and inputs_embeds.shape[1] > prefill_step_size
        ):
            total_toks = inputs_embeds.shape[1]
            with tqdm(total=total_toks, desc="Prefill", unit="tok") as pbar:
                while inputs_embeds.shape[1] > 1:
                    n_to_process = min(
                        prefill_step_size, inputs_embeds.shape[1] - 1
                    )
                    model.language_model(
                        inputs=input_ids[:, :n_to_process],
                        inputs_embeds=inputs_embeds[:, :n_to_process],
                        cache=prompt_cache,
                        n_to_process=n_to_process,
                        **kwargs,
                    )
                    quantize_cache_fn(prompt_cache)
                    mx.eval([c.state for c in prompt_cache])
                    inputs_embeds = inputs_embeds[:, n_to_process:]
                    input_ids = input_ids[:, n_to_process:]
                    mx.clear_cache()
                    pbar.update(n_to_process)
            input_ids = input_ids[:, -1:]

        # First step: process last token of prompt
        y, logprobs = _step_single(input_ids)

    mx.async_eval(y)

    # === Generation Phase with Speculative Decoding ===
    # V5 Design: Async pipeline + batch verification + zero-cost checkpoint.
    #
    # Profiling results (Qwen3.5-9B, M2 Ultra):
    #   Single token forward: 17.08ms
    #   Batch(5) forward:     33.28ms (2.57x speedup vs 5×single)
    #   Checkpoint:            0.01ms (zero-cost reference save)
    #   Restore:               0.04ms
    #   Reprocess:            18.42ms
    #
    # Strategy: standard async pipeline for common path (zero overhead),
    # batch verify on N-gram match (amortizes weight reads across tokens).
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
        if thinking_budget_criteria is not None:
            y = thinking_budget_criteria.apply_forced_token(y)
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
            # next_y is already being computed. Check if it matches draft[0].
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
                    # Only first draft or budget exhausted
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

                # Save cache checkpoint (zero-cost: reference save only)
                checkpoint = save_cache_checkpoint(prompt_cache)

                # Batch forward: [next_tid, remaining_0, ..., remaining_N]
                verify_input = mx.array(
                    [[next_tid] + list(remaining)], dtype=input_ids.dtype
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
                        dtype=input_ids.dtype,
                    )
                    with mx.stream(generation_stream):
                        model.language_model(
                            reprocess, cache=prompt_cache, **kwargs
                        )
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


def speculative_stream_generate(
    model: nn.Module,
    processor: PreTrainedTokenizer,
    prompt: str,
    image: Union[str, List[str]] = None,
    audio: Union[str, List[str]] = None,
    *,
    ngram_size: int = 3,
    max_draft_tokens: int = 5,
    **kwargs,
) -> Generator[SpeculativeGenerationResult, None, None]:
    """
    A generator producing text using prompt lookup speculative decoding.

    Args:
        model: The VLM model.
        processor: The tokenizer/processor.
        prompt: The input prompt text.
        image: Image path(s) or URL(s).
        audio: Audio file path(s).
        ngram_size: Size of N-grams for lookup (default: 3).
        max_draft_tokens: Maximum draft tokens per step (default: 5).
        **kwargs: Additional options passed to speculative_generate_step.

    Yields:
        SpeculativeGenerationResult with text, statistics, and metadata.
    """
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    # Handle thinking budget
    thinking_budget = kwargs.pop("thinking_budget", None)
    thinking_end_token = kwargs.pop("thinking_end_token", "</think>")
    thinking_start_token = kwargs.pop("thinking_start_token", None)
    enable_thinking = kwargs.pop("enable_thinking", False)

    skip_special_tokens = kwargs.pop("skip_special_tokens", False)
    skip_special_token_ids = (
        set(tokenizer.all_special_ids)
        if skip_special_tokens and hasattr(tokenizer, "all_special_ids")
        else []
    )

    add_special_tokens = (
        not hasattr(processor, "chat_template")
        if model.config.model_type in ["gemma3", "gemma3n"]
        else True
    )

    resize_shape = kwargs.pop("resize_shape", None)
    image_token_index = getattr(model.config, "image_token_index", None)

    if kwargs.get("input_ids", None) is not None:
        input_ids = kwargs.pop("input_ids")
        pixel_values = kwargs.pop("pixel_values", None)
        mask = kwargs.pop("mask", None)
    else:
        inputs = prepare_inputs(
            processor,
            images=image,
            audio=audio,
            prompts=prompt,
            image_token_index=image_token_index,
            resize_shape=resize_shape,
            add_special_tokens=add_special_tokens,
            **kwargs,
        )
        input_ids = inputs.get("input_ids", None)
        pixel_values = inputs.get("pixel_values", None)
        mask = inputs.get("attention_mask", None)
        data_kwargs = {
            k: v
            for k, v in inputs.items()
            if k not in ["input_ids", "pixel_values", "attention_mask"]
        }
        kwargs.update(data_kwargs)

    if thinking_budget is not None:
        thinking_start_token_id = tokenizer.encode(
            thinking_start_token, add_special_tokens=False
        )[-1]
        enable_thinking = enable_thinking and (
            thinking_start_token_id in input_ids.flatten().tolist()
        )
        tokenizer.thinking_budget_criteria = ThinkingBudgetCriteria(
            tokenizer=tokenizer,
            thinking_budget=thinking_budget,
            thinking_end_token=thinking_end_token,
            thinking_start_token=thinking_start_token,
            enable_thinking=enable_thinking,
        )
        kwargs["thinking_budget_criteria"] = tokenizer.thinking_budget_criteria
    else:
        tokenizer.thinking_budget_criteria = None

    with wired_limit(model, [generation_stream]):
        detokenizer = processor.detokenizer
        detokenizer.reset()
        thinking_criteria = getattr(tokenizer, "thinking_budget_criteria", None)

        gen = speculative_generate_step(
            input_ids,
            model,
            pixel_values,
            mask,
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

            if thinking_criteria is not None:
                thinking_criteria(token)

            if tokenizer.stopping_criteria(token):
                break

            detokenizer.add_token(
                token, skip_special_token_ids=skip_special_token_ids
            )

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


def speculative_generate(
    model: nn.Module,
    processor: PreTrainedTokenizer,
    prompt: str,
    image: Union[str, List[str]] = None,
    audio: Union[str, List[str]] = None,
    verbose: bool = False,
    *,
    ngram_size: int = 3,
    max_draft_tokens: int = 5,
    **kwargs,
) -> SpeculativeGenerationResult:
    """
    Generate text using prompt lookup speculative decoding.

    Args:
        model: The VLM model.
        processor: The tokenizer/processor.
        prompt: The input prompt text.
        verbose: Print tokens and timing info.
        ngram_size: N-gram size for draft lookup.
        max_draft_tokens: Max draft tokens per speculative step.

    Returns:
        SpeculativeGenerationResult with full generation statistics.
    """
    if verbose:
        print("=" * 10)
        files = []
        if image is not None:
            files.extend(image)
        if audio is not None:
            files.extend(audio)
        if kwargs.get("video") is not None:
            files.extend(kwargs.get("video"))
        if files:
            print(f"Files: {files}", "\n")
        print("Prompt:", prompt)
        print(f"[Speculative] ngram_size={ngram_size}, max_draft={max_draft_tokens}")

    text = ""
    last_response = None

    eos_tokens = kwargs.get("eos_tokens", None)
    stopping_criteria = kwargs.get("stopping_criteria", None)

    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    if eos_tokens is not None:
        tokenizer.stopping_criteria.add_eos_token_ids(eos_tokens)
    elif stopping_criteria is not None:
        if isinstance(stopping_criteria, StoppingCriteria) or callable(
            stopping_criteria
        ):
            tokenizer.stopping_criteria = stopping_criteria
        else:
            raise ValueError(
                "stopping_criteria must be an instance of StoppingCriteria or a callable"
            )
    else:
        tokenizer.stopping_criteria.reset(model.config.eos_token_id)

    for response in speculative_stream_generate(
        model,
        processor,
        prompt,
        image,
        audio,
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
        if len(text) == 0:
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

    return SpeculativeGenerationResult(
        text=text,
        token=last_response.token if last_response else None,
        logprobs=last_response.logprobs if last_response else None,
        from_draft=last_response.from_draft if last_response else False,
        prompt_tokens=last_response.prompt_tokens if last_response else 0,
        generation_tokens=last_response.generation_tokens if last_response else 0,
        total_tokens=last_response.total_tokens if last_response else 0,
        prompt_tps=last_response.prompt_tps if last_response else 0,
        generation_tps=last_response.generation_tps if last_response else 0,
        peak_memory=last_response.peak_memory if last_response else 0,
        acceptance_rate=last_response.acceptance_rate if last_response else 0,
        tokens_per_step=last_response.tokens_per_step if last_response else 0,
    )
