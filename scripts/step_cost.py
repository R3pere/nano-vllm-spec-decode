"""How much does checking k guesses cost compared with a plain decode step, at each batch size?

For B requests with context length L, time ModelRunner.run for
  plain   1 token per request (what decoding without speculation does)
  verify  1+k tokens per request (the last token plus k guesses)
and print verify / plain. That ratio is the break-even point: speculative decoding is faster only if a step
produces more tokens per request on average than the ratio (P1 measured ~4 on the copy workload and ~1.2 on
ordinary chat). It ignores everything outside the model runner (scheduling, n-gram lookup, postprocess).

--spec draft: a draft model makes the guesses (default: the model guesses for itself). Then "verify" includes
the k drafting steps, "plain" includes the draft model's 2-token step that keeps its KV cache up to date (it
runs on steps that don't guess too), and an extra column times the k drafting steps alone.

The requests are made up (constant token ids). Their KV blocks are real, but when B requests need more
blocks than the cache has, later requests reuse earlier requests' blocks; the amount of KV read per step
stays right, it just comes from fewer distinct addresses.

Usage: MODEL=... python scripts/step_cost.py [--k 4] [--lengths 256 1024] [--temperature 0] [--enforce-eager]
                                             [--spec ngram|draft] [--draft-model PATH]
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
    p.add_argument("--spec", choices=["ngram", "draft"], default="ngram")
    p.add_argument("--draft-model", help="--spec draft: the model that guesses (default: --model itself)")
    args = p.parse_args()

    draft = args.spec == "draft"
    llm = LLM(args.model, enforce_eager=args.enforce_eager, speculative_method=args.spec, num_speculative_tokens=args.k,
              draft_model=(args.draft_model or args.model) if draft else None)
    runner = llm.model_runner
    block_size, num_blocks = runner.block_size, runner.config.num_kvcache_blocks
    sp = SamplingParams(temperature=args.temperature)

    def timed(fn):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(args.repeat):
            fn()
        torch.cuda.synchronize()
        return 1000 * (time.perf_counter() - t) / args.repeat

    def time_run(seqs, k):
        def run():
            for seq in seqs:
                seq.draft_token_ids = [7] * k if not draft else []
            runner.run(seqs, False, k)
        return timed(run)

    print(f"{'eager' if args.enforce_eager else 'CUDA graph'}, {args.spec}, k={args.k}, temperature {args.temperature}, "
          f"verify batches up to {llm.spec_max_batch_size} requests, {num_blocks} KV blocks")
    for L in args.lengths:
        blocks_per_seq = (L + args.k + block_size - 1) // block_size
        num_sets = num_blocks // blocks_per_seq
        print(f"\ncontext length {L} ({num_sets} requests fit in the KV cache)")
        print(f"{'requests':>8s} {'plain ms':>9s} {'verify ms':>9s} {'ratio':>6s}" + (f" {'drafting ms':>11s}" if draft else ""))
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
            plain = time_run(seqs, 0)
            verify = time_run(seqs, args.k)
            line = f"{B:8d} {plain:9.2f} {verify:9.2f} {verify / plain:6.2f}"
            if draft:
                temperatures = runner.prepare_sample(seqs)
                line += f" {timed(lambda: runner.run_draft(seqs, args.k, temperatures)):11.2f}"
            print(line, flush=True)


if __name__ == "__main__":
    main()
