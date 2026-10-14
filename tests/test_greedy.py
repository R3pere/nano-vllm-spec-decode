"""Greedy correctness test.

Why not just compare tokens one by one?
  The model runs in bfloat16, which only keeps ~3 significant digits. Logits around 20
  are rounded to steps of 0.125, so two candidate tokens are often *exactly* tied, and
  any change in computation order (different kernel, batch size, chunking) can flip
  which one wins. After one flip the two texts go different ways. Even HuggingFace in
  bf16 vs HuggingFace in fp32 disagree this way. So "every token identical" is not
  something any two bf16 implementations can promise.

What we check instead (per generated token, no matter which variant produced it):
  Feed prompt + generated text into HuggingFace in float32 (the referee) and look at
  the logits at each generated position. The chosen token must be the referee's top-1,
  or within TOL of the top-1 logit (i.e. a near-tie that bf16 noise can flip).
  A real bug (wrong KV cache, wrong position, wrong block) produces tokens far from
  top-1 within a few steps, so it fails loudly.

Variants checked:
  hf_bf16        - HuggingFace bf16, hand-written greedy loop (shows what bf16 noise looks like)
  eager          - nano-vllm without CUDA Graph
  graph          - nano-vllm with CUDA Graph
  graph_chunked  - nano-vllm with CUDA Graph and a 512-token budget per step,
                   which forces chunked prefill and prefix-cache hits
  spec_ngram     - nano-vllm eager with n-gram speculative decoding (k=4)
  spec_ngram_graph - the same with CUDA Graph (checking guesses also runs as a graph)
  spec_ngram_graph_chunked - the same with the 512-token budget
  spec_draft     - nano-vllm eager, a draft model guesses (k=4); the model guesses for itself here
                   (two copies of the same model: 1.7B + 0.6B don't fit in 6GB)
  spec_draft_graph, spec_draft_graph_chunked - the same with CUDA Graph / with the 512-token budget
  spec_ngram_dynk_graph, spec_draft_dynk_graph - CUDA Graph, k chosen every step (at most 8)
Each runs in its own process so they don't fight over GPU memory.

Pairs that run exactly the same computation, only eager vs CUDA Graph, must give identical
tokens on every prompt (MUST_MATCH): graph = eager, spec_ngram_graph = spec_ngram,
spec_draft_graph = spec_draft. (Dynamic k picks k from timings, so runs can differ in k and in
bf16 near-ties; they only face the referee.)

A model guessing for itself should have almost every guess accepted (MIN_ACCEPTANCE). A bug in the
draft model's KV cache doesn't change the output (the big model checks every guess), only this rate.
The rate checked is accepted / (accepted + rejections), the chance that a tested guess is kept. Not
accepted / proposed: that one falls as k grows (after one bf16 flip the rest of the step's guesses all
count as not accepted), e.g. 0.969 at k=4 but 0.942 at k=8 for the same per-guess rate ~0.9875.

Usage: python tests/test_greedy.py [--model PATH] [--max-tokens N] [--tol T] [--variants NAME ...]
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

VARIANTS = {
    "hf_bf16": None,
    "eager": dict(enforce_eager=True),
    "graph": dict(enforce_eager=False),
    "graph_chunked": dict(enforce_eager=False, max_num_batched_tokens=512),
    "spec_ngram": dict(enforce_eager=True, speculative_method="ngram", num_speculative_tokens=4),
    "spec_ngram_graph": dict(enforce_eager=False, speculative_method="ngram", num_speculative_tokens=4),
    "spec_ngram_graph_chunked": dict(enforce_eager=False, max_num_batched_tokens=512, speculative_method="ngram", num_speculative_tokens=4),
    # draft_model=SELF: the model under test guesses for itself
    "spec_draft": dict(enforce_eager=True, speculative_method="draft", draft_model="SELF", num_speculative_tokens=4),
    "spec_draft_graph": dict(enforce_eager=False, speculative_method="draft", draft_model="SELF", num_speculative_tokens=4),
    "spec_draft_graph_chunked": dict(enforce_eager=False, max_num_batched_tokens=512, speculative_method="draft", draft_model="SELF", num_speculative_tokens=4),
    "spec_ngram_dynk_graph": dict(enforce_eager=False, speculative_method="ngram", num_speculative_tokens=8, spec_dynamic_k=True),
    "spec_draft_dynk_graph": dict(enforce_eager=False, speculative_method="draft", draft_model="SELF", num_speculative_tokens=8, spec_dynamic_k=True),
}
MUST_MATCH = {"graph": "eager", "spec_ngram_graph": "spec_ngram", "spec_draft_graph": "spec_draft"}
MIN_ACCEPTANCE = {name: 0.95 for name in VARIANTS if name.startswith("spec_draft")}

QUESTIONS = [
    "introduce yourself",
    "list all prime numbers within 100",
    "Write a short poem about the sea.",
    "Explain what a hash table is in two sentences.",
    "What is 17 * 23? Show your steps.",
    "Translate 'good morning, how are you' into French and German.",
]

# A long shared prefix (> 1 KV block of 256 tokens) so later prompts can reuse earlier prompts' KV cache.
LONG_CONTEXT = " ".join(f"Fact {i}: {i} squared is {i * i}, and {i} plus seven is {i + 7}." for i in range(1, 41))
CONTEXT_QUESTIONS = [
    "What is 12 squared according to the facts?",
    "Which fact mentions the number 30?",
    "Sum the squares of facts 1 to 3.",
    "Is there a fact about 50?",
]


def build_prompts(model):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model)

    def chat(text):
        return tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True)

    prompts = [chat(q) for q in QUESTIONS]
    prompts += [chat(f"{LONG_CONTEXT}\n\nQuestion: {q}") for q in CONTEXT_QUESTIONS]
    return prompts, tokenizer.eos_token_id


def load_hf(model, dtype):
    import torch
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(model, dtype=getattr(torch, dtype)).cuda().eval()


def run_hf_greedy(model, prompts, max_tokens, eos):
    import torch
    hf = load_hf(model, "bfloat16")
    outputs = []
    with torch.inference_mode():
        for ids in prompts:
            x = torch.tensor([ids], device="cuda")
            past = None
            out = []
            for _ in range(max_tokens):
                r = hf(input_ids=x, past_key_values=past, use_cache=True)
                past = r.past_key_values
                token = r.logits[0, -1].argmax().item()
                out.append(token)
                if token == eos:
                    break
                x = torch.tensor([[token]], device="cuda")
            outputs.append(out)
    return outputs


def run_nanovllm(model, prompts, max_tokens, **kwargs):
    from nanovllm import LLM, SamplingParams
    if kwargs.get("draft_model") == "SELF":
        kwargs = dict(kwargs, draft_model=model)
    llm = LLM(model, **kwargs)
    outputs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=max_tokens), use_tqdm=False)
    acceptance = None
    if llm.spec_method:
        s = llm.get_spec_stats()
        acceptance = s["guess_acceptance"]
        print(f"  acceptance rate {s['acceptance_rate']:.3f} (per tested guess {s['guess_acceptance']:.3f}), tokens per step {s['tokens_per_step']:.2f}, "
              f"steps per k {s['k_steps']}, KV blocks {llm.config.num_kvcache_blocks}", flush=True)
    return {"tokens": [o["token_ids"] for o in outputs], "acceptance": acceptance}


def referee(model, prompts, results):
    """For every generated token, how far its fp32 logit is below the fp32 top-1 logit (0 = it is top-1)."""
    import torch
    hf = load_hf(model, "float32")
    gaps = {}
    with torch.inference_mode():
        for name, outputs in results.items():
            gaps[name] = []
            for prompt, out in zip(prompts, outputs):
                x = torch.tensor([prompt + out], device="cuda")
                logits = hf(input_ids=x).logits[0, len(prompt) - 1:-1].float()    # logits that predicted each output token
                chosen = logits.gather(1, torch.tensor(out, device="cuda").unsqueeze(1)).squeeze(1)
                gaps[name].append((logits.max(dim=1).values - chosen).tolist())
    return gaps


def worker(args):
    prompts, eos = build_prompts(args.model)
    if args.worker == "referee":
        with open(args.inp) as f:
            result = referee(args.model, prompts, json.load(f))
    elif args.worker == "hf_bf16":
        result = {"tokens": run_hf_greedy(args.model, prompts, args.max_tokens, eos), "acceptance": None}
    else:
        result = run_nanovllm(args.model, prompts, args.max_tokens, **VARIANTS[args.worker])
    with open(args.out, "w") as f:
        json.dump(result, f)


def call_worker(name, out, args, inp=None):
    cmd = [sys.executable, __file__, "--worker", name, "--out", out, "--model", args.model, "--max-tokens", str(args.max_tokens)]
    if inp:
        cmd += ["--inp", inp]
    print(f"running {name} ...", flush=True)
    subprocess.run(cmd, check=True)
    with open(out) as f:
        return json.load(f)


def first_diff(a, b):
    n = min(len(a), len(b))
    return next((j for j in range(n) if a[j] != b[j]), n if len(a) != len(b) else None)


def main(args):
    with tempfile.TemporaryDirectory() as tmp:
        # eager is the reference for "same as eager"; MUST_MATCH partners are needed for the exact check
        wanted = set(args.variants) | {"eager"} | {MUST_MATCH[n] for n in args.variants if n in MUST_MATCH}
        names = [n for n in VARIANTS if n in wanted]
        runs = {name: call_worker(name, os.path.join(tmp, f"{name}.json"), args) for name in names}
        results = {name: run["tokens"] for name, run in runs.items()}
        with open(os.path.join(tmp, "all.json"), "w") as f:
            json.dump(results, f)
        gaps = call_worker("referee", os.path.join(tmp, "gaps.json"), args, inp=os.path.join(tmp, "all.json"))

    num_prompts = len(results["eager"])
    print(f"\n{num_prompts} prompts, tolerance {args.tol}")
    print(f"{'variant':24s} {'tokens':>6s} {'not top-1':>9s} {'worst gap':>9s} {'same as eager':>13s}  verdict")
    all_ok = True
    for name in names:
        flat = [g for seq in gaps[name] for g in seq]
        not_top1 = sum(g > 0 for g in flat)
        worst = max(flat)
        same = sum(first_diff(a, b) is None for a, b in zip(results["eager"], results[name]))
        ok = worst <= args.tol
        note = ""
        if name in MUST_MATCH:
            partner = MUST_MATCH[name]
            same_partner = sum(first_diff(a, b) is None for a, b in zip(results[partner], results[name]))
            note = f"  (same as {partner}: {same_partner}/{num_prompts}, must be all)"
            ok &= same_partner == num_prompts
        if name in MIN_ACCEPTANCE:
            acceptance = runs[name]["acceptance"]
            note += f"  (acceptance per tested guess {acceptance:.3f}, must be >= {MIN_ACCEPTANCE[name]})"
            ok &= acceptance >= MIN_ACCEPTANCE[name]
        all_ok &= ok
        print(f"{name:24s} {len(flat):6d} {not_top1:9d} {worst:9.4f} {same:>6d}/{num_prompts:<6d}  {'OK' if ok else 'FAIL'}{note}")
        if worst > args.tol:
            for i, seq in enumerate(gaps[name]):
                bad = [(j, g) for j, g in enumerate(seq) if g > args.tol]
                if bad:
                    print(f"    prompt {i}: {len(bad)} tokens over tolerance, first at token {bad[0][0]} (gap {bad[0][1]:.4f})")
    print("PASS" if all_ok else "FAIL")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.environ.get("MODEL", os.path.expanduser("~/huggingface/Qwen3-0.6B/")))
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--tol", type=float, default=0.5, help="max allowed fp32 logit gap below top-1")
    p.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    p.add_argument("--worker", choices=list(VARIANTS) + ["referee"])
    p.add_argument("--inp")
    p.add_argument("--out")
    args = p.parse_args()
    worker(args) if args.worker else main(args)
