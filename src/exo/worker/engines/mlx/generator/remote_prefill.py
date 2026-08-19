import time
from collections.abc import Callable
from typing import cast

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, KVCache, RotatingKVCache

from exo.worker.disaggregated.protocol import Header, KVChunk
from exo.worker.disaggregated.server import PrefillRequest
from exo.worker.engines.mlx.cache import CacheSnapshot, snapshot_ssm_states
from exo.worker.engines.mlx.disaggregated.client import (
    ingest_into_mlx_cache,
    remote_prefill_fetch,
)
from exo.worker.engines.mlx.types import KVCacheType
from exo.worker.runner.bootstrap import logger


def remote_prefill(
    prompt_tokens: mx.array,
    cache: KVCacheType,
    on_prefill_progress: Callable[[int, int], None] | None,
    *,
    endpoint: str,
    request_id: str,
    model_id: str,
    start_pos: int = 0,
    vision_embeddings: mx.array | None = None,
    vision_image_token_id: int | None = None,
    image_hashes: list[str] | None = None,
    raw_images_base64: list[str] | None = None,
) -> tuple[float, int, list[CacheSnapshot]]:
    t0 = time.perf_counter()
    total_prompt_tokens = int(prompt_tokens.shape[0])
    num_layers: int = 0

    def _on_header(header: Header) -> None:
        nonlocal num_layers
        num_layers = header.num_layers

    def _on_chunk(_chunk: KVChunk, chunks_received: int) -> None:
        nonlocal num_layers
        if on_prefill_progress is None:
            return
        if num_layers > 0 and chunks_received % num_layers == 0:
            tokens_so_far = chunks_received // num_layers
            on_prefill_progress(
                min(tokens_so_far, total_prompt_tokens),
                total_prompt_tokens,
            )

    vision_embeddings_bytes: bytes | None = None
    vision_embeddings_shape: list[int] | None = None
    vision_embeddings_dtype: str | None = None
    if vision_embeddings is not None:
        import numpy as np

        emb_f16 = vision_embeddings.astype(mx.float16)
        emb_np = np.array(emb_f16)
        vision_embeddings_bytes = emb_np.tobytes()
        vision_embeddings_shape = list(vision_embeddings.shape)
        vision_embeddings_dtype = "float16"

    safe_image_hashes: list[str] | None = (
        [str(h) for h in image_hashes] if image_hashes is not None else None
    )
    safe_raw_images: list[str] | None = (
        [str(img) for img in raw_images_base64]
        if raw_images_base64 is not None
        else None
    )

    request = PrefillRequest(
        model_id=model_id,
        token_ids=cast(list[int], prompt_tokens.tolist()),
        start_pos=start_pos,
        request_id=request_id,
        vision_embeddings_bytes=vision_embeddings_bytes,
        vision_embeddings_shape=vision_embeddings_shape,
        vision_embeddings_dtype=vision_embeddings_dtype,
        vision_image_token_id=vision_image_token_id,
        image_hashes=safe_image_hashes,
        raw_images_base64=safe_raw_images,
    )
    result = remote_prefill_fetch(
        endpoint, request, on_header=_on_header, on_kv_chunk=_on_chunk
    )
    t_received = time.perf_counter()

    caches = cast(list[KVCache | RotatingKVCache | ArraysCache], list(cache))
    final_offset = ingest_into_mlx_cache(result, caches, start_pos=start_pos)

    # Materialize all injected cache arrays immediately to prevent Metal concatenation graph depth accumulation
    eval_arrays = []
    for c in caches:
        if hasattr(c, "keys") and getattr(c, "keys", None) is not None:
            eval_arrays.append(c.keys)
        if hasattr(c, "values") and getattr(c, "values", None) is not None:
            eval_arrays.append(c.values)
    if eval_arrays:
        mx.eval(*eval_arrays)

    t_done = time.perf_counter()

    num_tokens = final_offset - start_pos
    tps = num_tokens / max(t_done - t0, 0.001)

    logger.info(
        f"Remote prefill: {num_tokens} tokens (start_pos={start_pos}, "
        f"final_offset={final_offset}) at {tps:.0f} tok/s, "
        f"transfer={(t_received - t0) * 1000:.0f}ms, "
        f"inject={(t_done - t_received) * 1000:.0f}ms"
    )
    return tps, num_tokens, [snapshot_ssm_states(cache)]
