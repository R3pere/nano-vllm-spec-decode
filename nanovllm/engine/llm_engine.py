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
        self.scheduler = Scheduler(config)
        self.max_model_len = config.max_model_len
        self.num_spec_tokens = config.num_speculative_tokens
        self.spec_max_batch_size = config.spec_max_batch_size
        self.proposer = NgramProposer(config.prompt_lookup_min, config.prompt_lookup_max) if config.speculative_method == "ngram" else None
        self.reset_spec_stats()
        atexit.register(self.exit)

    def reset_spec_stats(self):
        self.spec_stats = dict(num_steps=0, num_drafts=0, num_accepted=0, num_emitted=0)

    def get_spec_stats(self) -> dict:
        """Speculative decoding counters since the last reset. A "step" here is one decode step of one request.
        acceptance_rate = accepted guesses / proposed guesses; tokens_per_step = tokens produced per decode step
        (plain decoding is 1.0)."""
        s = dict(self.spec_stats)
        s["acceptance_rate"] = s["num_accepted"] / max(s["num_drafts"], 1)
        s["tokens_per_step"] = s["num_emitted"] / max(s["num_steps"], 1)
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

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        # big batches don't guess: checking k guesses multiplies the work per step by 1+k, which stops paying
        # off once the GPU is busy, and there are no CUDA graphs for it
        if not is_prefill and self.proposer and len(seqs) <= self.spec_max_batch_size:
            for seq in seqs:
                # one step emits at most (guesses + 1) tokens, so don't guess past max_tokens
                k = min(self.num_spec_tokens, seq.max_tokens - seq.num_completion_tokens - 1)
                seq.draft_token_ids = self.proposer.propose(seq, k)
        num_prefill_tokens = sum(seq.num_scheduled_tokens for seq in seqs)
        num_drafts = sum(len(seq.draft_token_ids) for seq in seqs)
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        num_appended = self.scheduler.postprocess(seqs, token_ids, is_prefill)
        if not is_prefill and self.proposer:
            self.spec_stats["num_steps"] += len(seqs)
            self.spec_stats["num_drafts"] += num_drafts
            self.spec_stats["num_accepted"] += sum(len(t) - 1 for t in token_ids)
            self.spec_stats["num_emitted"] += num_appended
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
