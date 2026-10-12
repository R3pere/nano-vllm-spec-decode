"""Where does the time of one decode step go? Runs a workload at a fixed concurrency and splits every decode
step into phases (milliseconds per step, averaged over all decode steps):

  schedule      Scheduler.schedule (block reservation)
  ngram         drafting guesses on the CPU
  prepare       ModelRunner.prepare_decode (Python lists -> GPU tensors)
  model         the transformer itself (CUDA graph replay, or eager kernels)
  lm_head       logits (outside the graph, like upstream)
  verify        verify_drafts / Sampler (choosing the tokens)
  rest_of_run   the rest of ModelRunner.run: building the guess tensors, copying results back (.tolist())
  postprocess   Scheduler.postprocess (appending tokens, prefix-cache hashes)

Each phase ends with torch.cuda.synchronize() so GPU work is charged to the phase that launched it. That
removes the little CPU/GPU overlap there normally is, so the total is a bit above the real step time.

Usage: MODEL=... python scripts/profile_decode.py [--workload copy|chat] [--concurrency 4] [--num-seqs 16]
                                                   [--enforce-eager] [--spec ngram] [--k 4] [--temperature 0]
"""
import argparse
import os
import sys
import time
from collections import defaultdict
from random import Random

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import bench
from nanovllm import LLM, SamplingParams
from nanovllm.engine import model_runner as model_runner_module

PHASES = ["schedule", "ngram", "prepare", "model", "lm_head", "verify", "rest_of_run", "postprocess"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.environ.get("MODEL", os.path.expanduser("~/huggingface/Qwen3-0.6B/")))
    p.add_argument("--workload", choices=["copy", "chat"], default="copy")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--num-seqs", type=int, default=16)
    p.add_argument("--max-tokens", type=int, default=768)
    p.add_argument("--temperature", type=float, default=0)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--spec", choices=["ngram"])
    p.add_argument("--k", type=int, default=4)
    args = p.parse_args()

    llm = LLM(args.model, enforce_eager=args.enforce_eager, speculative_method=args.spec, num_speculative_tokens=args.k)
    runner, sched = llm.model_runner, llm.scheduler
    step_times = defaultdict(float)    # phase -> seconds, current step
    totals = defaultdict(float)        # phase -> seconds, all decode steps

    def timed(name, fn):
        def wrapper(*a, **kw):
            torch.cuda.synchronize()
            t = time.perf_counter()
            out = fn(*a, **kw)
            torch.cuda.synchronize()
            step_times[name] += time.perf_counter() - t
            return out
        return wrapper

    sched.schedule = timed("schedule", sched.schedule)
    sched.postprocess = timed("postprocess", sched.postprocess)
    if llm.proposer:
        llm.proposer.propose = timed("ngram", llm.proposer.propose)
    runner.prepare_decode = timed("prepare", runner.prepare_decode)
    runner.run_model = timed("run_model", runner.run_model)
    runner.model.compute_logits = timed("lm_head", runner.model.compute_logits)
    runner.sampler = timed("verify", runner.sampler)
    model_runner_module.verify_drafts = timed("verify", model_runner_module.verify_drafts)
    runner.run = timed("run", runner.run)

    make_prompts = bench.copy_prompts if args.workload == "copy" else bench.chat_prompts
    prompts = make_prompts(llm.tokenizer, args.num_seqs, Random(0))
    sp = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)
    llm.generate(prompts[:2], SamplingParams(temperature=args.temperature, max_tokens=64), use_tqdm=False)    # warmup
    llm.reset_spec_stats()

    num_steps = num_tokens = 0
    pending, running = list(prompts), []
    start = time.perf_counter()
    while pending or not llm.is_finished():
        while pending and sum(not s.is_finished for s in running) < args.concurrency:
            running.append(llm.add_request(pending.pop(0), sp))
        step_times.clear()
        torch.cuda.synchronize()
        t = time.perf_counter()
        _, n = llm.step()
        torch.cuda.synchronize()
        if n < 0:    # decode step
            step_times["step"] = time.perf_counter() - t
            for name, v in step_times.items():
                totals[name] += v
            num_steps += 1
            num_tokens += -n
    duration = time.perf_counter() - start

    ms = {name: 1000 * v / num_steps for name, v in totals.items()}
    ms["model"] = ms.get("run_model", 0) - ms.get("lm_head", 0)
    ms["rest_of_run"] = ms.get("run", 0) - ms.get("prepare", 0) - ms.get("run_model", 0) - ms.get("verify", 0)
    label = ("spec " if args.spec else "") + ("eager" if args.enforce_eager else "graph")
    print(f"{label}, {args.workload}, concurrency {args.concurrency}: {num_steps} decode steps, "
          f"{num_tokens / num_steps:.2f} tokens per step (all requests), {num_tokens / duration:.0f} tok/s with syncs")
    print("ms per decode step: " + "  ".join(f"{name} {ms.get(name, 0):.2f}" for name in PHASES) + f"  | step total {ms['step']:.2f}")


if __name__ == "__main__":
    main()
