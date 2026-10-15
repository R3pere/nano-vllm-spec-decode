"""Exit 0 if DIR holds a complete HuggingFace model download: config, tokenizer and every weight shard
(the ones model.safetensors.index.json lists, or a single model.safetensors) with a readable header.

Usage: python scripts/check_model.py DIR
"""
import json
import os
import struct
import sys


def check(d):
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        if not os.path.isfile(os.path.join(d, name)):
            return f"missing {name}"
    index = os.path.join(d, "model.safetensors.index.json")
    if os.path.isfile(index):
        with open(index) as f:
            shards = sorted(set(json.load(f)["weight_map"].values()))
    else:
        shards = ["model.safetensors"]
    for shard in shards:
        path = os.path.join(d, shard)
        if not os.path.isfile(path):
            return f"missing {shard}"
        # a safetensors file starts with the header length, then a JSON header giving every tensor's byte range
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        end = max(v["data_offsets"][1] for k, v in header.items() if k != "__metadata__")
        if os.path.getsize(path) != 8 + n + end:
            return f"{shard} is truncated"
    return None


if __name__ == "__main__":
    error = check(sys.argv[1]) if os.path.isdir(sys.argv[1]) else "no such directory"
    if error:
        sys.exit(f"{sys.argv[1]}: {error}")
    print(f"{sys.argv[1]}: complete")
