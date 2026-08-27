import time
import mlx.core as mx
import numpy as np
from exo.worker.engines.mlx.disaggregated.adapter import quantize_kv_int8, dequantize_kv_int8, array_to_bytes

print("=== Testing KV Cache INT8 Wire Quantization ===")

# Simulate Qwen3.6-35B KV cache chunk: 2048 tokens, 8 heads, 128 head_dim
shape = (2048, 8, 128)
orig_tensor = mx.random.normal(shape).astype(mx.bfloat16)
mx.eval(orig_tensor)

raw_bytes = array_to_bytes(orig_tensor)
raw_size = len(raw_bytes)

t0 = time.perf_counter()
q_bytes, s_bytes = quantize_kv_int8(orig_tensor)
t_quant = (time.perf_counter() - t0) * 1000

compressed_size = len(q_bytes) + len(s_bytes)

t0 = time.perf_counter()
reconstructed = dequantize_kv_int8(q_bytes, s_bytes, shape, target_dtype=mx.bfloat16)
mx.eval(reconstructed)
t_dequant = (time.perf_counter() - t0) * 1000

# Numerical diff
diff = mx.abs(orig_tensor - reconstructed)
max_diff = float(mx.max(diff))
mean_diff = float(mx.mean(diff))

# Cosine similarity
orig_flat = orig_tensor.reshape(-1).astype(mx.float32)
recon_flat = reconstructed.reshape(-1).astype(mx.float32)
dot = float(mx.sum(orig_flat * recon_flat))
norm_orig = float(mx.sqrt(mx.sum(orig_flat * orig_flat)))
norm_recon = float(mx.sqrt(mx.sum(recon_flat * recon_flat)))
cos_sim = dot / (norm_orig * norm_recon)

print(f"Original 16-bit Size   : {raw_size / 1024 / 1024:.2f} MB")
print(f"Compressed INT8 Size   : {compressed_size / 1024 / 1024:.2f} MB ({compressed_size / raw_size * 100:.1f}%)")
print(f"Payload Size Reduction : {(1 - compressed_size / raw_size) * 100:.1f}%")
print(f"Quantize Time          : {t_quant:.2f} ms")
print(f"Dequantize Time        : {t_dequant:.2f} ms")
print(f"Max Absolute Error     : {max_diff:.5f}")
print(f"Mean Absolute Error    : {mean_diff:.5f}")
print(f"Cosine Similarity      : {cos_sim:.6f}")

assert cos_sim > 0.9995, f"Cosine similarity too low: {cos_sim}"
print("\n>>> ALL TESTS PASSED SUCCESSFULLY! <<<")
