"""vLLM behind the same small interface bench.py drives nano-vllm with (see NanoEngine there), so both engines
run the same workload through the same client loop and timing code, without an HTTP server in between.

The engine runs in this process (VLLM_ENABLE_V1_MULTIPROCESSING=0): one step() is one scheduling step of the
engine core plus turning its output into RequestOutputs, like LLMEngine.step in nano-vllm. Everything else
stays at vLLM's defaults (V1 engine, torch.compile + piecewise CUDA graphs, chunked prefill, async scheduling
if this version turns it on), and the result records them.

Kept the same as nano-vllm for a fair comparison: model, bf16, max_num_seqs, max_num_batched_tokens,
gpu_memory_utilization, max_model_len, prefix caching on, CUDA graphs on (unless --enforce-eager), the n-gram
lookup lengths (2..4) and k. Prompts go in as token ids and outputs are not detokenized.

Tested with vLLM 0.15.1 (the first release with standalone draft models in the V1 engine, PR #24322).
"""
import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

PROMPT_LOOKUP_MIN, PROMPT_LOOKUP_MAX = 2, 4    # nano-vllm's Config defaults


class Request:
    __slots__ = ("num_completion_tokens", "is_finished")

    def __init__(self):
        self.num_completion_tokens = 0
        self.is_finished = False


class VllmEngine:

    def __init__(self, args):
        import vllm
        from transformers import AutoTokenizer
        from vllm import EngineArgs, SamplingParams
        from vllm.inputs import TokensPrompt
        from vllm.v1.engine.llm_engine import LLMEngine
        self.SamplingParams, self.TokensPrompt = SamplingParams, TokensPrompt
        self.version = vllm.__version__
        self.draft_model = (args.draft_model or args.model) if args.spec == "draft" else None
        spec_config = None
        if args.spec == "ngram":
            spec_config = {"method": "ngram", "num_speculative_tokens": args.k,
                           "prompt_lookup_min": PROMPT_LOOKUP_MIN, "prompt_lookup_max": PROMPT_LOOKUP_MAX}
        elif args.spec == "draft":
            spec_config = {"method": "draft_model", "model": self.draft_model, "num_speculative_tokens": args.k}
        engine_args = EngineArgs(
            model=args.model, dtype="bfloat16", seed=args.seed, max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs, max_num_batched_tokens=args.max_num_batched_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization, enable_prefix_caching=True,
            enforce_eager=args.enforce_eager, speculative_config=spec_config, disable_log_stats=False,
        )
        self.engine = LLMEngine.from_engine_args(engine_args)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model)
        self.requests = {}
        self.next_id = 0
        self.spec = args.spec
        self.metrics_base = {}

    def add(self, prompt, temperature, max_tokens, ignore_eos):
        rid = str(self.next_id)
        self.next_id += 1
        sp = self.SamplingParams(temperature=temperature, max_tokens=max_tokens, ignore_eos=ignore_eos, detokenize=False)
        self.engine.add_request(rid, self.TokensPrompt(prompt_token_ids=prompt), sp)
        self.requests[rid] = req = Request()
        return req

    def step(self):
        for out in self.engine.step():
            req = self.requests[out.request_id]
            req.num_completion_tokens = len(out.outputs[0].token_ids)    # outputs are cumulative
            if out.finished:
                req.is_finished = True
                del self.requests[out.request_id]
        return "all"

    def counters(self):
        """vLLM's Prometheus counters (vllm:...) summed over labels; needs disable_log_stats=False."""
        from vllm.v1.metrics.reader import Counter
        totals = {}
        for m in self.engine.get_metrics():
            if isinstance(m, Counter):
                totals[m.name] = totals.get(m.name, 0) + m.value
        return totals

    def reset_stats(self):
        self.metrics_base = self.counters()

    def stats(self):
        config = self.engine.vllm_config
        s = {"engine_version": f"vllm {self.version}", "kv_blocks": config.cache_config.num_gpu_blocks,
             "kv_tokens": config.cache_config.num_gpu_blocks * config.cache_config.block_size,
             "vllm_config": {"async_scheduling": getattr(config.scheduler_config, "async_scheduling", None),
                             "chunked_prefill": config.scheduler_config.enable_chunked_prefill,
                             "prefix_caching": config.cache_config.enable_prefix_caching,
                             "cudagraph_mode": str(getattr(config.compilation_config, "cudagraph_mode", None)),
                             "compilation_mode": str(getattr(config.compilation_config, "mode", None) or getattr(config.compilation_config, "level", None)),
                             "attention_backend": os.environ.get("VLLM_ATTENTION_BACKEND")}}
        c = self.counters()
        delta = {name: value - self.metrics_base.get(name, 0) for name, value in c.items()}
        s["preemptions"] = int(delta.get("vllm:num_preemptions", 0))
        if self.spec:
            drafts = delta.get("vllm:spec_decode_num_draft_tokens", 0)
            accepted = delta.get("vllm:spec_decode_num_accepted_tokens", 0)
            num_drafts = delta.get("vllm:spec_decode_num_drafts", 0)    # decode steps of one request that guessed
            s["spec_stats"] = {"num_drafts": int(drafts), "num_accepted": int(accepted), "num_steps": int(num_drafts),
                               "acceptance_rate": accepted / max(drafts, 1),
                               # only steps that guessed (nano-vllm's tokens_per_step counts every decode step; compare
                               # the engine-independent tokens_per_arrival that bench.py measures instead)
                               "tokens_per_guessing_step": 1 + accepted / max(num_drafts, 1)}
        return s
