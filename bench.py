"""Benchmark: prints TTFT, TPOT and throughput.

  TTFT (time to first token)   = first token time - request arrival time
  TPOT (time per output token) = (finish time - first token time) / (output tokens - 1)
  throughput                   = output tokens / total wall time

Default workload is the same as upstream bench.py: 256 requests with random token
prompts (100-1024 tokens) and random output lengths (100-1024 tokens), all arriving
at once. Use --rate to make requests arrive over time (Poisson, requests/second).

Usage: python bench.py [--num-seqs N] [--rate R] [--enforce-eager] [--out result.json]
"""
import argparse
import json
import os
import time
from random import expovariate, randint, seed

import numpy as np

from nanovllm import LLM, SamplingParams


def summarize(values_s):
    ms = np.array(values_s) * 1000
    return {"mean": float(ms.mean()), "p50": float(np.percentile(ms, 50)), "p99": float(np.percentile(ms, 99))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.environ.get("MODEL", os.path.expanduser("~/huggingface/Qwen3-0.6B/")))
    p.add_argument("--num-seqs", type=int, default=256)
    p.add_argument("--input-len", type=int, nargs=2, default=[100, 1024], metavar=("MIN", "MAX"))
    p.add_argument("--output-len", type=int, nargs=2, default=[100, 1024], metavar=("MIN", "MAX"))
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--rate", type=float, default=0, help="requests per second; 0 = all arrive at once")
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", help="also write the result as JSON")
    args = p.parse_args()

    seed(args.seed)
    n = args.num_seqs
    prompts = [[randint(0, 10000) for _ in range(randint(*args.input_len))] for _ in range(n)]
    sampling_params = [SamplingParams(temperature=args.temperature, ignore_eos=True, max_tokens=randint(*args.output_len)) for _ in range(n)]
    arrivals = [0.0] * n
    if args.rate > 0:
        t = 0.0
        for i in range(n):
            arrivals[i] = t
            t += expovariate(args.rate)

    llm = LLM(args.model, enforce_eager=args.enforce_eager, max_model_len=args.max_model_len)
    llm.generate(["Benchmark: "], SamplingParams(), use_tqdm=False)    # warmup

    seqs = [None] * n
    first_token = [None] * n
    finish = [None] * n
    running = set()
    next_req = 0
    start = time.perf_counter()
    while next_req < n or running:
        now = time.perf_counter() - start
        while next_req < n and arrivals[next_req] <= now:
            seqs[next_req] = llm.add_request(prompts[next_req], sampling_params[next_req])
            running.add(next_req)
            next_req += 1
        if not running:    # idle until the next request arrives
            time.sleep(arrivals[next_req] - now)
            continue
        llm.step()
        now = time.perf_counter() - start
        for i in list(running):
            if first_token[i] is None and seqs[i].num_completion_tokens > 0:
                first_token[i] = now
            if seqs[i].is_finished:
                finish[i] = now
                running.remove(i)
    duration = time.perf_counter() - start

    input_tokens = sum(map(len, prompts))
    output_tokens = sum(sp.max_tokens for sp in sampling_params)
    result = {
        "num_seqs": n,
        "rate": args.rate,
        "enforce_eager": args.enforce_eager,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "duration_s": duration,
        "output_throughput": output_tokens / duration,
        "total_throughput": (input_tokens + output_tokens) / duration,
        "ttft_ms": summarize([first_token[i] - arrivals[i] for i in range(n)]),
        "tpot_ms": summarize([(finish[i] - first_token[i]) / max(sampling_params[i].max_tokens - 1, 1) for i in range(n)]),
    }

    print(f"requests: {n}  input tokens: {input_tokens}  output tokens: {output_tokens}  time: {duration:.2f}s")
    print(f"throughput: {result['output_throughput']:.1f} output tok/s, {result['total_throughput']:.1f} total tok/s")
    for name in ("ttft_ms", "tpot_ms"):
        s = result[name]
        print(f"{name[:4].upper()} (ms): mean {s['mean']:.1f}  p50 {s['p50']:.1f}  p99 {s['p99']:.1f}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
