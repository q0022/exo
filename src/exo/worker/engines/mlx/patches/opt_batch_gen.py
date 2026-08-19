from dataclasses import dataclass, field
from typing import cast

import mlx.core as mx
from mlx_lm.generate import GenerationBatch

_PRECOMPUTE_TOP_K = 20


@dataclass
class BatchTopKLogprobs:
    uids: list[int] = field(default_factory=list)
    indices: mx.array | None = None
    values: mx.array | None = None
    selected: mx.array | None = None
    _uid_to_row: dict[int, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._uid_to_row = {uid: i for i, uid in enumerate(self.uids)}

    def for_uid(self, uid: int) -> tuple[list[int], list[float], float] | None:
        if self.indices is None or self.values is None or self.selected is None:
            return None
        row = self._uid_to_row.get(uid)
        if row is None:
            return None
        return (
            cast(list[int], self.indices[row].tolist()),
            cast(list[float], self.values[row].tolist()),
            float(self.selected[row].item()),
        )


@dataclass
class _TopKBuffer:
    needs_topk: bool = False
    pending: BatchTopKLogprobs = field(default_factory=BatchTopKLogprobs)
    ready: BatchTopKLogprobs = field(default_factory=BatchTopKLogprobs)


def _get_buffer(batch: GenerationBatch) -> _TopKBuffer:
    buf = getattr(batch, "_topk_buffer", None)
    if buf is None:
        buf = _TopKBuffer()
        batch._topk_buffer = buf  # pyright: ignore[reportAttributeAccessIssue]
    return buf


def set_needs_topk(batch: GenerationBatch, needed: bool) -> None:
    _get_buffer(batch).needs_topk = needed


def take_ready_topk(batch: GenerationBatch) -> BatchTopKLogprobs:
    return _get_buffer(batch).ready


def _patched_step(self: GenerationBatch) -> tuple[list[int], list[mx.array]]:
    self._current_tokens = self._next_tokens
    self._current_logprobs = self._next_logprobs
    inputs = self._current_tokens
    assert inputs is not None, "_step requires initialized _next_tokens"

    buf = _get_buffer(self)
    buf.ready = buf.pending
    buf.pending = BatchTopKLogprobs()

    logits = self.model(inputs[:, None], cache=self.prompt_cache)
    logits = logits[:, -1, :]
    mx.eval(logits)

    if self.logits_processors is not None and any(self.logits_processors):
        processed_logits: list[mx.array] = []
        for e in range(len(self.uids)):
            sample_logits = logits[e : e + 1]
            for processor in self.logits_processors[e]:
                sample_logits = processor(mx.array(self.tokens[e]), sample_logits)
            processed_logits.append(sample_logits)
        logits = mx.concatenate(processed_logits, axis=0)
        mx.eval(logits)

    # Proactively drain Metal computation graph and clear cache every 10 steps
    if not hasattr(self, "_metal_step_count"):
        self._metal_step_count = 0
    self._metal_step_count += 1
    if self._metal_step_count % 10 == 0:
        mx.synchronize()
        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        if hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()

    if buf.needs_topk:
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        if self.samplers is not None and any(self.samplers):
            all_samples: list[mx.array] = []
            for e in range(len(self.uids)):
                sample_sampler = self.samplers[e] or self.fallback_sampler
                all_samples.append(sample_sampler(logprobs[e : e + 1]))
            sampled = mx.concatenate(all_samples, axis=0)
        else:
            sampled = self.fallback_sampler(logprobs)

        self._next_tokens = sampled
        self._next_logprobs = logprobs

        batch_size = len(self.uids)
        k = min(_PRECOMPUTE_TOP_K, logprobs.shape[1])
        pending_indices = mx.argpartition(-logprobs, k, axis=1)[:, :k]
        pending_values = mx.take_along_axis(logprobs, pending_indices, axis=1)
        sort_order = mx.argsort(-pending_values, axis=1)
        pending_indices = mx.take_along_axis(pending_indices, sort_order, axis=1)
        pending_values = mx.take_along_axis(pending_values, sort_order, axis=1)
        pending_selected = logprobs[mx.arange(batch_size), sampled]
        buf.pending = BatchTopKLogprobs(
            uids=list(self.uids),
            indices=pending_indices,
            values=pending_values,
            selected=pending_selected,
        )
        try:
            mx.eval(
                self._next_tokens,
                self._next_logprobs,
                pending_indices,
                pending_values,
                pending_selected,
            )
        except RuntimeError as e:
            if "metal::malloc" in str(e) or "Resource limit" in str(e):
                mx.synchronize()
                mx.clear_cache()
                import gc

                gc.collect()
                mx.eval(
                    self._next_tokens,
                    self._next_logprobs,
                    pending_indices,
                    pending_values,
                    pending_selected,
                )
            else:
                raise
    else:
        if self.samplers is not None and any(self.samplers):
            all_samples = []
            for e in range(len(self.uids)):
                sample_sampler = self.samplers[e] or self.fallback_sampler
                all_samples.append(sample_sampler(logits[e : e + 1]))
            sampled = mx.concatenate(all_samples, axis=0)
        else:
            sampled = self.fallback_sampler(logits)

        self._next_tokens = sampled
        self._next_logprobs = None

        # Synchronously evaluate token immediately to commit Metal command buffer and free 40-layer graph
        try:
            mx.eval(self._next_tokens)
        except RuntimeError as e:
            if "metal::malloc" in str(e) or "Resource limit" in str(e):
                mx.synchronize()
                mx.clear_cache()
                import gc

                gc.collect()
                mx.eval(self._next_tokens)
            else:
                raise

    current_lp = self._current_logprobs
    if current_lp is not None:
        try:
            if isinstance(current_lp, mx.array):
                mx.eval(inputs, current_lp)
            elif current_lp:
                mx.eval(inputs, *current_lp)
        except RuntimeError:
            mx.synchronize()
            mx.clear_cache()

    token_list = cast(list[int], inputs.tolist())
    for sti, ti in zip(self.tokens, token_list, strict=True):
        sti.append(ti)

    if isinstance(current_lp, mx.array):
        current_lp = list(current_lp)
    elif current_lp and len(current_lp) == len(token_list):
        current_lp = list(current_lp)
    else:
        current_lp = [None] * len(token_list)
    return token_list, current_lp


def _patched_filter(self: GenerationBatch, keep: list[int]) -> None:
    """Filter the batch to keep only the specified indices."""
    self.uids = [self.uids[idx] for idx in keep]
    if not keep:
        self.prompt_cache.clear()
    else:
        for c in self.prompt_cache:
            c.filter(keep)
    self.tokens = [self.tokens[idx] for idx in keep]
    if self.samplers is not None and any(self.samplers):
        self.samplers = [self.samplers[idx] for idx in keep]
    if hasattr(self, "logits_processors") and self.logits_processors is not None and any(self.logits_processors):
        self.logits_processors = [self.logits_processors[idx] for idx in keep]
    self.max_tokens = [self.max_tokens[idx] for idx in keep]
    self.state_machines = [self.state_machines[idx] for idx in keep]

    self._next_tokens = self._next_tokens[keep] if (keep and self._next_tokens is not None) else None
    if isinstance(self._next_logprobs, mx.array):
        self._next_logprobs = (
            self._next_logprobs[keep] if keep else None
        )
    elif isinstance(self._next_logprobs, list):
        self._next_logprobs = [self._next_logprobs[idx] for idx in keep]
    else:
        self._next_logprobs = None
    self._token_context = [self._token_context[idx] for idx in keep]
    self._num_tokens = [self._num_tokens[idx] for idx in keep]
    self._matcher_states = [self._matcher_states[idx] for idx in keep]


def apply_batch_gen_patch() -> None:
    GenerationBatch._step = _patched_step
    GenerationBatch.filter = _patched_filter
