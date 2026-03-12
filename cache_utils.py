"""
Cache checkpoint/restore utilities for speculative decoding with Qwen3.5.

Qwen3.5 uses a mixed cache architecture:
- ArraysCache (linear attention / GatedDeltaNet layers): NOT trimmable
- KVCache (full attention layers): trimmable via offset decrement

For speculative decoding, we need to "undo" cache updates when draft tokens
are rejected. This module provides zero-cost checkpoint/restore:
- ArraysCache: save Python references (model creates NEW arrays on update,
  so old references remain valid — no deep copy needed).
- KVCache: save the offset integer (restore = decrement offset).
"""

from typing import Any, List

from mlx_lm.models.cache import ArraysCache, KVCache, QuantizedKVCache, CacheList


def save_cache_checkpoint(cache: List[Any]) -> List[Any]:
    """
    Save a checkpoint of the entire prompt cache for speculative decoding.

    This is essentially free — it only saves Python references and integers.
    It works because GatedDeltaNet creates NEW arrays on each forward step
    (via mx.concatenate for conv_state and gated_delta_update for rec_state),
    so the old array objects are never modified in-place.

    Args:
        cache: The model's prompt cache (list of cache objects, one per layer).

    Returns:
        A list of checkpoint data that can be passed to restore_cache_checkpoint.
    """
    checkpoint = []
    for c in cache:
        if isinstance(c, ArraysCache):
            checkpoint.append(("arrays", list(c.cache)))
        elif isinstance(c, KVCache):
            checkpoint.append(("kv", c.offset))
        elif isinstance(c, QuantizedKVCache):
            checkpoint.append(("qkv", c.offset))
        elif isinstance(c, CacheList):
            sub = []
            for sub_c in c.caches:
                if isinstance(sub_c, ArraysCache):
                    sub.append(("arrays", list(sub_c.cache)))
                elif isinstance(sub_c, KVCache):
                    sub.append(("kv", sub_c.offset))
                else:
                    sub.append(("unknown", None))
            checkpoint.append(("cachelist", sub))
        else:
            checkpoint.append(("unknown", None))
    return checkpoint


def restore_cache_checkpoint(cache: List[Any], checkpoint: List[Any]) -> None:
    """
    Restore a cache to a previously saved checkpoint.

    For KVCache: restores the offset.
    For ArraysCache: restores the saved array references.

    Args:
        cache: The model's prompt cache.
        checkpoint: Checkpoint data from save_cache_checkpoint.
    """
    for c, cp in zip(cache, checkpoint):
        kind = cp[0]
        data = cp[1]
        if kind == "arrays":
            c.cache = data
        elif kind == "kv":
            c.offset = data
        elif kind == "qkv":
            c.offset = data
        elif kind == "cachelist":
            for sub_c, sub_cp in zip(c.caches, data):
                sub_kind = sub_cp[0]
                sub_data = sub_cp[1]
                if sub_kind == "arrays":
                    sub_c.cache = sub_data
                elif sub_kind == "kv":
                    sub_c.offset = sub_data
