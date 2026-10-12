"""Benchmark: prints TTFT, TPOT and throughput.

  TTFT (time to first token)   = first token time - request arrival time
  TPOT (time per output token) = (finish time - first token time) / (output tokens - 1)
  throughput                   = output tokens / total wall time

Workloads (--workload):
  random  same as upstream bench.py: random token prompts (100-1024 tokens), random output lengths
          (100-1024 tokens), EOS ignored. Note: on random prompts the model's output quickly falls
          into repeating a few tokens, so n-gram guessing works well here (acceptance ~0.8). It is
          NOT a "nothing to copy" case.
  copy    real editing tasks whose answer mostly repeats the input: "add comments to this code",
          "add type hints", "add docstrings", "fix typos in this text". The material is upstream
          nano-vllm's own source and README at commit bb823b3 (read with git, so it never changes).
          Output ends at EOS or --max-tokens.

All requests arrive at once unless --rate is given (Poisson arrivals, requests/second).
--concurrency C keeps at most C requests in flight: a new request is sent only when one finishes,
and its TTFT is counted from when it is sent. Speculative decoding helps most when few requests
run together, so test several levels.

Usage: python bench.py [--workload random|copy] [--num-seqs N] [--concurrency C] [--rate R]
                       [--temperature T] [--enforce-eager] [--spec ngram [--k K]] [--out result.json]
"""
import argparse
import json
import os
import subprocess
import time
from random import Random, expovariate, randint, seed

import numpy as np

from nanovllm import LLM, SamplingParams

ROOT = os.path.dirname(os.path.abspath(__file__))
MATERIAL_COMMIT = "bb823b3"    # upstream nano-vllm
CODE_FILES = [
    "nanovllm/engine/block_manager.py", "nanovllm/engine/llm_engine.py", "nanovllm/engine/model_runner.py",
    "nanovllm/engine/scheduler.py", "nanovllm/engine/sequence.py", "nanovllm/layers/attention.py",
    "nanovllm/layers/embed_head.py", "nanovllm/layers/layernorm.py", "nanovllm/layers/linear.py",
    "nanovllm/layers/rotary_embedding.py", "nanovllm/models/qwen3.py", "nanovllm/utils/loader.py", "bench.py",
]
CODE_TASKS = [
    "Add a short comment above every function and class in the following Python code. Keep everything else exactly the same and output the complete code.",
    "Add type hints to every function parameter and return value in the following Python code where they are missing. Output the complete modified code and nothing else.",
    "Add a one-line docstring to every function in the following Python code that does not have one. Output the complete code and nothing else.",
]
TEXT_TASK = "Fix any spelling or grammar mistakes in the following Markdown text and output the full corrected text, keeping all formatting and links."


def read_material(path):
    return subprocess.run(["git", "show", f"{MATERIAL_COMMIT}:{path}"], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def copy_prompts(tokenizer, n, rng):
    tasks = []
    for path in CODE_FILES:
        lines = read_material(path).splitlines()
        for i in range(0, len(lines), 40):
            chunk = "\n".join(lines[i:i + 40])
            if chunk.count("\n") >= 10:
                tasks += [f"{task}\n\n```python\n{chunk}\n```" for task in CODE_TASKS]
    readme = read_material("README.md").split("## Star History")[0]
    tasks.append(f"{TEXT_TASK}\n\n{readme}")
    rng.shuffle(tasks)
    tasks = [tasks[i % len(tasks)] for i in range(n)]
    return [tokenizer.apply_chat_template([{"role": "user", "content": t}], tokenize=True, add_generation_prompt=True, enable_thinking=False) for t in tasks]


def summarize(values_s):
    ms = np.array(values_s) * 1000
    return {"mean": float(ms.mean()), "p50": float(np.percentile(ms, 50)), "p99": float(np.percentile(ms, 99))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.environ.get("MODEL", os.path.expanduser("~/huggingface/Qwen3-0.6B/")))
    p.add_argument("--workload", choices=["random", "copy"], default="random")
    p.add_argument("--num-seqs", type=int, default=256)
    p.add_argument("--input-len", type=int, nargs=2, default=[100, 1024], metavar=("MIN", "MAX"), help="random workload")
    p.add_argument("--output-len", type=int, nargs=2, default=[100, 1024], metavar=("MIN", "MAX"), help="random workload")
    p.add_argument("--max-tokens", type=int, default=768, help="copy workload: output limit")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--rate", type=float, default=0, help="requests per second; 0 = all arrive at once")
    p.add_argument("--concurrency", type=int, default=0, help="most requests in flight; 0 = no limit")
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--spec", choices=["ngram"], help="speculative decoding method (default: off)")
    p.add_argument("--k", type=int, default=4, help="guessed tokens checked per step")
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", help="also write the result as JSON")
    args = p.parse_args()

    llm = LLM(args.model, enforce_eager=args.enforce_eager, max_model_len=args.max_model_len,
              speculative_method=args.spec, num_speculative_tokens=args.k)

    seed(args.seed)
    n = args.num_seqs
    if args.workload == "random":
        prompts = [[randint(0, 10000) for _ in range(randint(*args.input_len))] for _ in range(n)]
        sampling_params = [SamplingParams(temperature=args.temperature, ignore_eos=True, max_tokens=randint(*args.output_len)) for _ in range(n)]
    else:
        prompts = copy_prompts(llm.tokenizer, n, Random(args.seed))
        sampling_params = [SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)] * n
    arrivals = [0.0] * n
    if args.rate > 0:
        t = 0.0
        for i in range(n):
            arrivals[i] = t
            t += expovariate(args.rate)

    # Warmup. First calls with new shapes are slow (torch.compile of the sampler, kernel setup), so warm up
    # with one and with several requests. Repetitive made-up prompts make the model repeat itself, so n-gram
    # guesses get verified too; they share no text with the real prompts, so the prefix cache can't help those.
    warmup = [[100 + i % (3 + j) for i in range(64 * (j + 1))] for j in range(4)]
    llm.generate(warmup[:1], SamplingParams(temperature=args.temperature, max_tokens=32), use_tqdm=False)
    llm.generate(warmup, SamplingParams(temperature=args.temperature, max_tokens=32), use_tqdm=False)
    llm.reset_spec_stats()

    seqs = [None] * n
    first_token = [None] * n
    finish = [None] * n
    running = set()
    next_req = 0
    start = time.perf_counter()
    while next_req < n or running:
        now = time.perf_counter() - start
        while next_req < n and arrivals[next_req] <= now and not (args.concurrency and len(running) >= args.concurrency):
            arrivals[next_req] = max(arrivals[next_req], now)    # with a concurrency limit, a request "arrives" when it can be sent
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
    output_lens = [seq.num_completion_tokens for seq in seqs]
    output_tokens = sum(output_lens)
    result = {
        "workload": args.workload,
        "num_seqs": n,
        "concurrency": args.concurrency,
        "rate": args.rate,
        "temperature": args.temperature,
        "enforce_eager": args.enforce_eager,
        "spec": args.spec,
        "k": args.k if args.spec else 0,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "duration_s": duration,
        "output_throughput": output_tokens / duration,
        "total_throughput": (input_tokens + output_tokens) / duration,
        "ttft_ms": summarize([first_token[i] - arrivals[i] for i in range(n)]),
        "tpot_ms": summarize([(finish[i] - first_token[i]) / max(output_lens[i] - 1, 1) for i in range(n)]),
    }
    if args.spec:
        result["spec_stats"] = llm.get_spec_stats()

    print(f"requests: {n}  input tokens: {input_tokens}  output tokens: {output_tokens}  time: {duration:.2f}s")
    print(f"throughput: {result['output_throughput']:.1f} output tok/s, {result['total_throughput']:.1f} total tok/s")
    for name in ("ttft_ms", "tpot_ms"):
        s = result[name]
        print(f"{name[:4].upper()} (ms): mean {s['mean']:.1f}  p50 {s['p50']:.1f}  p99 {s['p99']:.1f}")
    if args.spec:
        s = result["spec_stats"]
        print(f"speculative: acceptance rate {s['acceptance_rate']:.3f}  tokens per step {s['tokens_per_step']:.2f}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
