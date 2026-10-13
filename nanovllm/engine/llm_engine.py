import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.ngram_proposer import NgramProposer
from nanovllm.engine import dynamic_k


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.config = config
        self.scheduler = Scheduler(config)
        self.max_model_len = config.max_model_len
        self.spec_method = config.speculative_method
        self.num_spec_tokens = config.num_speculative_tokens
        self.spec_max_batch_size = config.spec_batch_limit(self.num_spec_tokens) if self.spec_method else 0
        self.proposer = NgramProposer(config.prompt_lookup_min, config.prompt_lookup_max) if config.speculative_method == "ngram" else None
        # dynamic k: how long a step takes for every k and batch size, measured once (see dynamic_k.py)
        self.step_costs = self.step_cost_buckets = None
        if self.spec_method and config.spec_dynamic_k:
            self.step_costs, self.step_cost_buckets = self.model_runner.call("measure_step_costs")
        self.num_idle_steps = 0    # dynamic k, draft model: steps in a row that chose k = 0 (see dynamic_k.probe)
        self.model_runner.warn_if_low_memory()
        self.reset_spec_stats()
        atexit.register(self.exit)

    def reset_spec_stats(self):
        self.spec_stats = dict(num_steps=0, num_drafts=0, num_accepted=0, num_emitted=0, k_steps={})

    def get_spec_stats(self) -> dict:
        """Speculative decoding counters since the last reset. A "step" here is one decode step of one request.
        acceptance_rate = accepted guesses / proposed guesses; tokens_per_step = tokens produced per decode step
        (plain decoding is 1.0); k_steps[k] = how many engine steps used k (0 = didn't guess)."""
        s = dict(self.spec_stats)
        s["acceptance_rate"] = s["num_accepted"] / max(s["num_drafts"], 1)
        s["tokens_per_step"] = s["num_emitted"] / max(s["num_steps"], 1)
        s["k_steps"] = dict(sorted(s["k_steps"].items()))
        return s

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        assert len(prompt) < self.max_model_len, f"prompt has {len(prompt)} tokens, max_model_len is {self.max_model_len}"
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)
        return seq

    def choose_num_spec_tokens(self, seqs: list[Sequence]) -> int:
        """k for this decode step (0 = don't guess). With n-gram, also fills in seq.draft_token_ids, and
        seq.unchecked_guess when it found guesses but k = 0."""
        for seq in seqs:
            seq.unchecked_guess = None
        # big batches don't guess: checking k guesses multiplies the work per step by 1+k, which stops paying
        # off once the GPU is busy, and there are no CUDA graphs for it
        ks = [k for k in self.config.spec_k_choices if k == 0 or len(seqs) <= self.config.spec_batch_limit(k)]
        if not any(ks):
            return 0
        max_drafts = [seq.max_drafts(max(ks)) for seq in seqs]
        if self.proposer:
            # n-gram guesses are free to make, so make them first: how many each request found tells how many it can use
            for seq, m in zip(seqs, max_drafts):
                seq.draft_token_ids = self.proposer.propose(seq, m)
            max_drafts = [len(seq.draft_token_ids) for seq in seqs]
        if not any(max_drafts):
            return 0
        if self.step_costs is None:    # fixed k
            return max(ks)
        alphas = [dynamic_k.acceptance(seq.spec_accepted, seq.spec_rejected) for seq in seqs]
        costs = {k: self.step_costs[k] for k in ks}
        k = dynamic_k.choose_k(alphas, max_drafts, costs, self.step_cost_buckets)
        if self.proposer:
            for seq in seqs:
                if k == 0 and seq.draft_token_ids:
                    seq.unchecked_guess = seq.draft_token_ids[0]
                seq.draft_token_ids = seq.draft_token_ids[:k]
        else:
            # a draft model's guesses can't be checked for free like n-gram ones (see step), so probe now and then
            k, self.num_idle_steps = dynamic_k.probe(k, ks, self.num_idle_steps)
        return k

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        k = self.choose_num_spec_tokens(seqs) if not is_prefill and self.spec_method else 0
        num_prefill_tokens = sum(seq.num_scheduled_tokens for seq in seqs)
        token_ids = self.model_runner.call("run", seqs, is_prefill, k)
        if not is_prefill and self.spec_method:
            # the guesses checked this step (a draft model makes them inside run); postprocess clears them
            num_drafts = [len(seq.draft_token_ids) for seq in seqs]
            for seq, m, t in zip(seqs, num_drafts, token_ids):
                if m == 0 and seq.unchecked_guess is not None:
                    # n-gram guess not checked (k = 0): the model's own token equals it exactly when checking it
                    # would have accepted it. Greedy: both mean "it is the argmax". Sampling: it happens with
                    # probability p(guess), which is exactly the chance the check accepts an n-gram guess. So this
                    # is a free, unbiased observation of one guess, and alpha doesn't get stuck where k = 0 wins.
                    seq.spec_accepted, seq.spec_rejected = dynamic_k.update_counts(seq.spec_accepted, seq.spec_rejected, 1, int(t[0] == seq.unchecked_guess))
                else:
                    seq.spec_accepted, seq.spec_rejected = dynamic_k.update_counts(seq.spec_accepted, seq.spec_rejected, m, len(t) - 1)
        num_appended = self.scheduler.postprocess(seqs, token_ids, is_prefill)
        if not is_prefill and self.spec_method:
            self.spec_stats["num_steps"] += len(seqs)
            self.spec_stats["num_drafts"] += sum(num_drafts)
            self.spec_stats["num_accepted"] += sum(len(t) - 1 for t in token_ids)
            self.spec_stats["num_emitted"] += num_appended
            self.spec_stats["k_steps"][k] = self.spec_stats["k_steps"].get(k, 0) + 1
        for seq in seqs:
            if seq.is_finished and self.proposer:
                self.proposer.free(seq)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        # positive: prefill tokens processed; negative: decode tokens produced
        num_tokens = num_prefill_tokens if is_prefill else -num_appended
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
