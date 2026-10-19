"""Does padding a decode batch to the CUDA graph bucket change the real rows' results? (P5b2)

For each real batch size B whose bucket (nanovllm.engine.model_runner.batch_buckets) is bigger, computes the same
B rows alone and padded to the bucket, and checks they are bitwise equal: flash-attn decode (paged KV cache, as in
Attention.forward) and the GEMMs of Qwen3-0.6B / 1.7B / 8B. Random weights and inputs; only the shapes matter.
"same" means eager and graph can only agree if eager pads its batch the way graphs do.

Usage: python scripts/check_padding.py [--max-batch 24]
"""
import argparse
import os
import sys

import torch
from flash_attn import flash_attn_with_kvcache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nanovllm.engine.model_runner import batch_buckets

GEMMS = [("0.6B qkv_proj", 1024, 4096), ("0.6B down_proj", 3072, 1024), ("1.7B qkv_proj", 2048, 4096), ("1.7B down_proj", 6144, 2048),
         ("8B qkv_proj", 4096, 6144), ("8B gate_up_proj", 4096, 24576), ("8B down_proj", 12288, 4096)]


def compare(alone, padded):
    return "same" if torch.equal(alone, padded) else "DIFF"


def main(max_batch):
    torch.manual_seed(0)
    buckets = batch_buckets(512)
    sizes = [(b, next(x for x in buckets if x >= b)) for b in range(1, max_batch + 1)]
    sizes = [(b, p) for b, p in sizes if p != b]
    props = torch.cuda.get_device_properties(0)
    print(f"{props.name}, {props.multi_processor_count} SMs; real batch -> padded batch")
    page, width, ctx, head_dim = 256, 17, 300, 128    # width 17 = ceil(4096 / 256), like the graphs' block tables
    for num_heads, num_kv_heads, name in [(16, 8, "0.6B / 1.7B"), (32, 8, "8B")]:
        k_cache = torch.randn(64, page, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
        v_cache = torch.randn_like(k_cache)
        out = []
        for b, p in sizes:
            q = torch.randn(p, 1, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
            block_tables = torch.full((p, width), -1, dtype=torch.int32, device="cuda")
            block_tables[:b, :2] = torch.randint(0, 64, (b, 2), dtype=torch.int32, device="cuda")
            lens = torch.tensor([ctx] * b + [0] * (p - b), dtype=torch.int32, device="cuda")    # padding: empty requests
            alone = flash_attn_with_kvcache(q[:b], k_cache, v_cache, cache_seqlens=lens[:b], block_table=block_tables[:b], causal=True)
            padded = flash_attn_with_kvcache(q, k_cache, v_cache, cache_seqlens=lens, block_table=block_tables, causal=True)[:b]
            out.append(f"{b}->{p} {compare(alone, padded)}")
        print(f"attention {name} ({num_heads} q heads, {num_kv_heads} kv heads): " + ", ".join(out))
    for name, k, n in GEMMS:
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
        out = []
        for b, p in sizes:
            x = torch.randn(p, k, device="cuda", dtype=torch.bfloat16)
            out.append(f"{b}->{p} {compare(torch.nn.functional.linear(x[:b], w), torch.nn.functional.linear(x, w)[:b])}")
        print(f"GEMM {name} [{k} -> {n}]: " + ", ".join(out))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--max-batch", type=int, default=24)
    main(p.parse_args().max_batch)
