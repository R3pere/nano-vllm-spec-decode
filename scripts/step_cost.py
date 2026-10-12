"""How much does checking k guesses cost compared with a plain decode step, at each batch size?

For B requests with context length L, time ModelRunner.run for
  plain   1 token per request (what decoding without speculation does)
  verify  1+k tokens per request (the last token plus k guesses)
and print verify / plain. That ratio is the break-even point: speculative decoding is faster only if a step
produces more tokens per request on average than the ratio (P1 measured ~4 on the copy workload and ~1.2 on
ordinary chat). It ignores everything outside the model runner (scheduling, n-gram lookup, postprocess).

The requests are made up (constant token ids). Their KV blocks are real, but when B requests need more
blocks than the cache has, later requests reuse earlier requests' blocks; the amount of KV read per step
stays right, it just comes from fewer distinct addresses.

Usage: MODEL=... python scripts/step_cost.py [--k 4] [--lengths 256 1024] [--temperature 0] [--enforce-eager]
"""
import argparse
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from nanovllm import LLM, SamplingParams
from nanovllm.engine.sequence import Sequence


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.environ.get("MODEL", os.path.expanduser("~/huggingface/Qwen3-0.6B/")))
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--lengths", type=int, nargs="+", default=[256, 1024])
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 48, 64, 96])
    p.add_argument("--temperature", type=float, default=0)
    p.add_argument("--repeat", type=int, default=30)
    p.add_argument("--enforce-eager", action="store_true")
    args = p.parse_args()

    llm = LLM(args.model, enforce_eager=args.enforce_eager, speculative_method="ngram", num_speculative_tokens=args.k)
    runner = llm.model_runner
    block_size, num_blocks = runner.block_size, runner.config.num_kvcache_blocks
    sp = SamplingParams(temperature=args.temperature)

    def time_run(seqs, drafts):
        for seq in seqs:
            seq.draft_token_ids = list(drafts)
        for _ in range(3):
            runner.run(seqs, False)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(args.repeat):
            runner.run(seqs, False)
        torch.cuda.synchronize()
        return 1000 * (time.perf_counter() - t) / args.repeat

    print(f"{'eager' if args.enforce_eager else 'CUDA graph'}, k={args.k}, temperature {args.temperature}, "
          f"verify batches up to {llm.spec_max_batch_size} requests")
    for L in args.lengths:
        blocks_per_seq = (L + args.k + block_size - 1) // block_size
        num_sets = num_blocks // blocks_per_seq
        print(f"\ncontext length {L} ({num_sets} requests fit in the KV cache)")
        print(f"{'requests':>8s} {'plain ms':>9s} {'verify ms':>9s} {'ratio':>6s}")
        for B in args.batch_sizes:
            if B > llm.spec_max_batch_size:
                continue
            seqs = []
            for i in range(B):
                seq = Sequence([100 + i] * L, sp)
                j = i % num_sets
                seq.block_table = list(range(j * blocks_per_seq, (j + 1) * blocks_per_seq))
                seq.num_cached_tokens = L - 1
                seqs.append(seq)
            plain = time_run(seqs, [])
            verify = time_run(seqs, [7] * args.k)
            print(f"{B:8d} {plain:9.2f} {verify:9.2f} {verify / plain:6.2f}", flush=True)


if __name__ == "__main__":
    main()
