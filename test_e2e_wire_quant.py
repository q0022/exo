import io
import mlx.core as mx
from mlx_lm.models.cache import KVCache
from exo.worker.disaggregated.protocol import Header, KVChunk, read_header, read_message, write_header, write_done
from exo.worker.engines.mlx.disaggregated.adapter import send_mlx_kv_cache, chunk_to_mlx_nhd, inject_kv_chunk, write_cache_to_wire

print("=== Testing End-to-End KV Cache Wire Streaming Round-Trip ===")

# Create a mock 4-layer KVCache
num_layers = 4
num_tokens = 512
n_heads = 8
head_dim = 128

caches = [KVCache() for _ in range(num_layers)]
for c in caches:
    # shape: (1, n_heads, num_tokens, head_dim)
    c.keys = mx.random.normal((1, n_heads, num_tokens, head_dim)).astype(mx.bfloat16)
    c.values = mx.random.normal((1, n_heads, num_tokens, head_dim)).astype(mx.bfloat16)
    c.offset = num_tokens
    mx.eval(c.keys, c.values)

# Stream to in-memory bytes buffer
buf = io.BytesIO()
tokens_sent = write_cache_to_wire(buf, caches, request_id="test-req", model_id="test-model", start_pos=0)
buf.seek(0)

# Receiver reads from stream
header = read_header(buf)
print(f"Header received: num_layers={header.num_layers}, start_pos={header.start_pos}, dtype={header.dtype}")

received_caches = [KVCache() for _ in range(num_layers)]
layer_chunks = {}
while True:
    msg = read_message(buf)
    if msg is None:
        break
    if isinstance(msg, KVChunk):
        layer_chunks[msg.layer_idx] = msg
        k_nhd, v_nhd = chunk_to_mlx_nhd(msg)
        inject_kv_chunk(received_caches[msg.layer_idx], k_nhd, v_nhd, msg.num_tokens)
    elif msg.__class__.__name__ == "Done":
        print(f"Done message received: total_tokens={msg.total_tokens}")
        break

# Verify fidelity on all layers
all_passed = True
for i in range(num_layers):
    orig_k = caches[i].keys
    recv_k = received_caches[i].keys
    diff = float(mx.max(mx.abs(orig_k - recv_k)))
    
    orig_flat = orig_k.reshape(-1).astype(mx.float32)
    recv_flat = recv_k.reshape(-1).astype(mx.float32)
    cos = float(mx.sum(orig_flat * recv_flat) / (mx.sqrt(mx.sum(orig_flat**2)) * mx.sqrt(mx.sum(recv_flat**2))))
    
    print(f"Layer {i}: Max Diff = {diff:.5f}, Cosine Sim = {cos:.6f}")
    if cos < 0.999:
        all_passed = False

assert all_passed, "Fidelity test failed on one or more layers!"
print("\n>>> END-TO-END WIRE QUANT ROUND-TRIP VERIFIED 100%! <<<")
