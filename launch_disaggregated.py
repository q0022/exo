import argparse
import json
import time
import urllib.request
import urllib.error

API_BASE = "http://localhost:52415"

def post_json(endpoint, payload):
    req = urllib.request.Request(
        f"{API_BASE}{endpoint}",
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    try:
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as e:
        print(f"Error {e.code}: {e.read().decode()}")
        raise

def get_json(endpoint):
    req = urllib.request.Request(f"{API_BASE}{endpoint}", method='GET')
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode())

def get_active_instances(model_id, role=None):
    state = get_json("/state")
    instances = []
    for inst_id, raw_inst in state.get("instances", {}).items():
        # Handle enum wrapper
        inst = raw_inst.get("MlxRingInstance") or raw_inst.get("MlxJacclInstance") or raw_inst.get("VllmInstance") or raw_inst
        
        shard_assignments = inst.get("shardAssignments", {})
        if shard_assignments.get("modelId") == model_id:
            node_runner = shard_assignments.get("nodeToRunner", {})
            nodes = list(node_runner.keys())
            if not nodes:
                continue
            
            instances.append((inst_id, inst))
    return instances

def launch_disaggregated(model_id: str):
    print(f"Launching {model_id} in Disaggregated mode...")
    
    # 1. Launch Prefill (will go to DGX)
    print("Requesting Prefill instance...")
    post_json("/place_instance", {
        "model_id": model_id,
        "preferred_role": "prefill"
    })
    
    # 2. Launch Decode (will go to Mac)
    print("Requesting Decode instance...")
    post_json("/place_instance", {
        "model_id": model_id,
        "preferred_role": "decode"
    })
    
    # 3. Wait for both instances to appear
    print("Waiting for instances to be allocated...")
    prefill_id = None
    decode_id = None
    
    for _ in range(15):
        time.sleep(2)
        instances = get_active_instances(model_id)
        if len(instances) >= 2:
            state = get_json("/state")
            identities = state.get("nodeIdentities", {})
            
            for inst_id, inst in instances:
                nodes = list(inst.get("shardAssignments", {}).get("nodeToRunner", {}).keys())
                if not nodes:
                    continue
                node_id = nodes[0]
                identity = identities.get(node_id, {})
                os_version = identity.get("osVersion", "")
                chip_id = identity.get("chipId", "")
                
                # Simple heuristic: Linux = DGX (Prefill), Apple = Mac (Decode)
                is_dgx = "Linux" in os_version
                is_mac = "Apple" in chip_id or "Mac" in identity.get("friendlyName", "")
                
                if is_dgx and not prefill_id:
                    prefill_id = inst_id
                elif is_mac and not decode_id:
                    decode_id = inst_id
            
            if prefill_id and decode_id:
                break
    
    if not prefill_id or not decode_id:
        print("Failed to find allocated prefill and decode instances.")
        print(f"prefill_id: {prefill_id}, decode_id: {decode_id}")
        return
        
    print(f"Found Prefill ID: {prefill_id}")
    print(f"Found Decode ID: {decode_id}")
    
    # 4. Link them together
    print("Linking instances...")
    post_json("/v1/instance-links", {
        "prefill_instances": [prefill_id],
        "decode_instances": [decode_id]
    })
    print("Disaggregated cluster successfully setup!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch a model in disaggregated mode")
    parser.add_argument("model_id", type=str, help="Model ID (e.g., mlx-community/Qwen3.6-35B-A3B-4bit)")
    args = parser.parse_args()
    
    launch_disaggregated(args.model_id)
