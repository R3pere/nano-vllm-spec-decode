"""Where does the time of one mixed step go (mixed batching: decode requests and prefill chunks together)?
Sends --num-seqs long random-token prompts at once with a small token budget, so prompts are prefilled in chunks
while the earlier ones already decode, and splits every mixed step into phases (milliseconds per mixed step):

  schedule     Scheduler.schedule_mixed
  prepare      ModelRunner.prepare_mixed (the flat batch, Python lists -> GPU tensors)
  attention    the 28 attentions (store_kvcache + flash_attn_varlen_func), run eagerly in both modes
  rest_model   the rest of the transformer: piecewise graph replays (or eager kernels with --no-mixed-graph),
               plus copying inputs into the graph buffers
  lm_head      logits of the rows that need them
  sample       verify_drafts / Sampler, and the rest of run_mixed (.tolist())
  postprocess  Scheduler.postprocess

Each phase ends with torch.cuda.synchronize(), and so does every attention: that removes the CPU/GPU overlap
there normally is (the CPU launching the next kernels while the GPU still works), so the phases add up to more
than the step time without syncs, which is printed too (from a separate pass without syncs).

Usage: MODEL=... python scripts/profile_mixed.py [--budget 512] [--no-mixed-graph] [--enforce-eager] [--num-seqs 12]
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

from nanovllm import LLM, SamplingParams
from nanovllm.layers.attention import Attention

PHASES = ["schedule", "prepare", "attention", "rest_model", "lm_head", "sample", "postprocess"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.environ.get("MODEL", os.path.expanduser("~/huggingface/Qwen3-0.6B/")))
    p.add_argument("--budget", type=int, default=512, help="max_num_batched_tokens")
    p.add_argument("--no-mixed-graph", action="store_true", help="mixed steps eager, unpadded (P4a)")
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--num-seqs", type=int, default=12)
    p.add_argument("--max-tokens", type=int, default=128)
    args = p.parse_args()

    llm = LLM(args.model, enforce_eager=args.enforce_eager, enable_mixed_batching=True,
              mixed_cuda_graph=not args.no_mixed_graph, max_num_batched_tokens=args.budget)
    runner, sched = llm.model_runner, llm.scheduler
    sp = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=args.max_tokens)

    def make_prompts(seed):    # new random text for each pass, so the prefix cache never helps
        rng = Random(seed)
        return [[rng.randint(0, 10000) for _ in range(rng.randint(1024, 2048))] for _ in range(args.num_seqs)]

    llm.generate(make_prompts(2)[:2], SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=16), use_tqdm=False)    # warmup

    def run(with_syncs):
        """All prompts at once; returns (phase -> ms per mixed step, number of mixed steps)."""
        prompts = make_prompts(int(with_syncs))
        step_times, totals, num_steps = defaultdict(float), defaultdict(float), 0
        originals = []

        def timed(obj, name, phase):
            fn = getattr(obj, name)
            originals.append((obj, name, fn))

            def wrapper(*a, **kw):
                torch.cuda.synchronize()
                t = time.perf_counter()
                out = fn(*a, **kw)
                torch.cuda.synchronize()
                step_times[phase] += time.perf_counter() - t
                return out
            setattr(obj, name, wrapper)

        if with_syncs:
            timed(sched, "schedule_mixed", "schedule")
            timed(sched, "postprocess", "postprocess")
            timed(runner, "prepare_mixed", "prepare")
            timed(runner, "run_piecewise", "model")
            timed(runner, "run_model", "model")
            timed(runner.model, "compute_logits", "lm_head")
            timed(runner, "run_mixed", "run")
            timed(Attention, "forward", "attention")
        for prompt in prompts:
            llm.add_request(prompt, sp)
        while not llm.is_finished():
            step_times.clear()
            num_mixed = llm.num_mixed_steps
            torch.cuda.synchronize()
            t = time.perf_counter()
            llm.step()
            torch.cuda.synchronize()
            if llm.num_mixed_steps > num_mixed:
                step_times["step"] = time.perf_counter() - t
                for name, v in step_times.items():
                    totals[name] += v
                num_steps += 1
        for obj, name, fn in originals:
            setattr(obj, name, fn)
        ms = {name: 1000 * v / num_steps for name, v in totals.items()}
        return ms, num_steps

    ms, num_steps = run(True)    # first, so that the pass without syncs runs on shapes seen before
    plain, num_plain = run(False)
    ms["rest_model"] = ms.get("model", 0) - ms.get("attention", 0)
    ms["sample"] = ms.get("run", 0) - sum(ms.get(x, 0) for x in ("prepare", "model", "lm_head"))
    label = "eager" if args.enforce_eager else "P4a: mixed steps eager" if args.no_mixed_graph else "piecewise graphs"
    print(f"{label}, budget {args.budget}: {num_steps} mixed steps; step without syncs {plain['step']:.2f} ms ({num_plain} mixed steps)")
    print("ms per mixed step (with syncs): " + "  ".join(f"{name} {ms.get(name, 0):.2f}" for name in PHASES) + f"  | total {ms['step']:.2f}")


if __name__ == "__main__":
    main()
