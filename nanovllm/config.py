import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    speculative_method: str | None = None    # None (off) or "ngram"
    num_speculative_tokens: int = 4          # k: at most k guessed tokens are checked per step
    prompt_lookup_min: int = 2               # ngram: shortest tail to look up in earlier text
    prompt_lookup_max: int = 4               # ngram: longest tail to look up in earlier text

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert self.speculative_method in (None, "ngram")
        if self.speculative_method:
            # drafting needs each request's full token list, which TP>1 doesn't send to the worker processes
            assert self.tensor_parallel_size == 1
            assert self.num_speculative_tokens >= 1
            assert self.max_num_batched_tokens >= 1 + self.num_speculative_tokens
            assert 1 <= self.prompt_lookup_min <= self.prompt_lookup_max
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
