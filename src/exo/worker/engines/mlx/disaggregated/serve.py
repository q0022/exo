import contextlib
import time

import mlx.core as mx
import numpy as np
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.worker.disaggregated.server import PrefillRequest
from exo.worker.engines.mlx.cache import (
    KVPrefixCache,
    cache_length,
    make_kv_cache,
    snapshot_ssm_states,
)
from exo.worker.engines.mlx.disaggregated.vision_cache import global_vision_cache
from exo.worker.engines.mlx.generator.generate import patch_embed_tokens
from exo.worker.engines.mlx.generator.generate import prefill as mlx_prefill
from exo.worker.engines.mlx.types import KVCacheType, Model
from exo.worker.engines.mlx.utils_mlx import fix_unmatched_think_end_tokens
from exo.worker.runner.bootstrap import logger


def run_prefill_for_request(
    *,
    model: Model,
    tokenizer: TokenizerWrapper,
    group: mx.distributed.Group | None,
    kv_prefix_cache: KVPrefixCache | None,
    request: PrefillRequest,
) -> KVCacheType:
    prompt_tokens = mx.array(request.token_ids)
    prompt_tokens = fix_unmatched_think_end_tokens(prompt_tokens, tokenizer)
    n_tokens = int(prompt_tokens.shape[0])
    t0 = time.perf_counter()

    matched_index: int | None = None
    prefix_hit_length = 0
    if kv_prefix_cache is not None:
        cache, remaining, matched_index, _ = kv_prefix_cache.get_kv_cache(
            model, prompt_tokens
        )
        prefix_hit_length = n_tokens - int(remaining.shape[0])
    else:
        cache = make_kv_cache(model)
        remaining = prompt_tokens

    new_tokens = max(0, n_tokens - prefix_hit_length)
    if 0 < new_tokens < 4 and prefix_hit_length > 0:
        extra = 4 - new_tokens
        prefix_hit_length = max(0, prefix_hit_length - extra)
        new_tokens = max(0, n_tokens - prefix_hit_length)
    prefill_input = prompt_tokens[prefix_hit_length : prefix_hit_length + new_tokens]
    if 0 < int(prefill_input.shape[0]) < 4:
        pad_amount = 4 - int(prefill_input.shape[0])
        pad_tokens = mx.zeros((pad_amount,), dtype=prefill_input.dtype)
        prefill_input = mx.concatenate([prefill_input, pad_tokens])
        new_tokens = int(prefill_input.shape[0])

    maybe_vision_ctx = contextlib.nullcontext()
    vision_embeddings: mx.array | None = None
    vision_image_token_id: int | None = request.vision_image_token_id
    primary_hash: str | None = (
        request.image_hashes[0] if request.image_hashes else None
    )

    # 1. Content-Addressable Checksum Registry Lookup (Fast Path)
    if primary_hash:
        cached_result = global_vision_cache.get(primary_hash)
        if cached_result is not None:
            vision_embeddings, cached_token_id = cached_result
            if vision_image_token_id is None:
                vision_image_token_id = cached_token_id

    # 2. If Cache Miss, deserialize embeddings or compute from raw images
    if vision_embeddings is None:
        if (
            request.vision_embeddings_bytes is not None
            and request.vision_embeddings_shape is not None
        ):
            try:
                dtype_name = request.vision_embeddings_dtype or "float16"
                np_dtype = (
                    np.float16 if dtype_name in ("float16", "bfloat16") else np.float32
                )
                arr_np = np.frombuffer(
                    request.vision_embeddings_bytes, dtype=np_dtype
                ).reshape(request.vision_embeddings_shape)
                vision_embeddings = mx.array(arr_np)

                # Store in Vision Registry Cache for subsequent stages on the same image
                if primary_hash and vision_image_token_id is not None:
                    global_vision_cache.put(
                        primary_hash, vision_embeddings, vision_image_token_id
                    )
            except Exception:
                logger.opt(exception=True).warning(
                    "Failed to deserialize and patch vision embeddings on prefill server"
                )

    # 3. Patch embedding layer if vision embeddings are ready
    if vision_embeddings is not None and vision_image_token_id is not None:
        try:
            maybe_vision_ctx = patch_embed_tokens(
                model,
                vision_embeddings,
                start_offset=prefix_hit_length,
                token_count=new_tokens,
                image_token_id=vision_image_token_id,
            )
        except Exception:
            logger.opt(exception=True).warning(
                "Failed to patch vision embeddings into model"
            )

    if int(prefill_input.shape[0]) >= 4:
        sampler = make_sampler(temp=1.0)
        with maybe_vision_ctx:
            _ = mlx_prefill(
                model=model,
                tokenizer=tokenizer,
                sampler=sampler,
                prompt_tokens=prefill_input,
                cache=cache,
                group=group,
                on_prefill_progress=None,
                distributed_prompt_progress_callback=None,
            )

    if kv_prefix_cache is not None:
        try:
            cache_snapshots = [snapshot_ssm_states(cache)]
            hit_ratio = prefix_hit_length / n_tokens if n_tokens > 0 else 0.0
            if matched_index is not None and hit_ratio >= 0.5:
                kv_prefix_cache.update_kv_cache(
                    matched_index,
                    prompt_tokens,
                    cache,
                    cache_snapshots,
                    restore_pos=prefix_hit_length,
                )
            else:
                kv_prefix_cache.add_kv_cache(prompt_tokens, cache, cache_snapshots)
        except Exception:
            logger.opt(exception=True).warning(
                "Failed to save prefix cache on prefill server"
            )

    elapsed = time.perf_counter() - t0
    final_offset = cache_length(cache)
    logger.info(
        f"Prefill: request_id={request.request_id} "
        f"{n_tokens} tokens (prefix_hit={prefix_hit_length}, "
        f"final_offset={final_offset}) in {elapsed * 1000:.0f}ms"
    )
    return cache
