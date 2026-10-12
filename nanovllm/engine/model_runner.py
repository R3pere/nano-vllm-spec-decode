import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler, verify_drafts
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        # Batch sizes CUDA graphs are recorded for, keyed by tokens per sequence: 1 (plain decode) and, with
        # speculative decoding, 1+k (checking k guesses). Known even with enforce_eager, see prepare_decode.
        k = config.num_speculative_tokens if config.speculative_method else 0
        max_bs = {1: min(config.max_num_seqs, 512)}
        if k:
            max_bs[1 + k] = config.spec_max_batch_size
        self.graph_bs = {n: [x for x in [1, 2, 4, 8] if x < bs] + list(range(16, bs, 16)) + [bs] for n, bs in max_bs.items()}
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        if self.config.speculative_method:
            self.warmup_verify()
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def warmup_verify(self):
        # Verifying guesses produces logits for every query token, (1+k) rows per request, e.g. 510 x 151936
        # for 102 requests and k=4. The KV cache gets whatever memory is left after the peak seen during
        # warmup, so run the largest verify once here to include it in the peak.
        k = self.config.num_speculative_tokens
        num_seqs = self.config.spec_max_batch_size    # bigger batches don't guess (see LLMEngine.step)
        set_context(False)
        hidden_states = torch.zeros(num_seqs * (1 + k), self.config.hf_config.hidden_size)
        logits = self.model.compute_logits(hidden_states)
        drafts = torch.zeros(num_seqs, k, dtype=torch.int64)
        num_drafts = torch.full((num_seqs,), k, dtype=torch.int64)
        temperatures = torch.ones(num_seqs)
        verify_drafts(logits, drafts, num_drafts, temperatures)
        reset_context()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence], num_padding_seqs: int = 0):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables += [[-1] * max_len] * num_padding_seqs
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence], num_query_tokens: int = 1):
        # Each sequence brings num_query_tokens tokens: its last token (whose KV isn't cached yet), then its
        # guessed tokens, then padding. Padding is never accepted; its KV lands past the cached length and is
        # overwritten later. All sequences use the same count so attention can treat q as [seqs, count, ...].
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            tokens = [seq.last_token] + seq.draft_token_ids
            tokens += [tokens[-1]] * (num_query_tokens - len(tokens))
            start = len(seq) - 1
            end = start + num_query_tokens
            input_ids.extend(tokens)
            positions.extend(range(start, end))
            context_lens.append(end)
            slot_mapping.extend(seq.block_table[i // self.block_size] * self.block_size + i % self.block_size for i in range(start, end))
        # With speculative decoding, pad the batch with empty sequences up to a size CUDA graphs are recorded
        # for, also when running eagerly. cuBLAS and flash-attn results depend slightly on the batch size, so
        # this way eager and graph compute exactly the same thing and give the same tokens. Empty sequences
        # write no KV (slot -1) and see nothing (length 0); their outputs are dropped.
        num_padding = 0
        sizes = self.graph_bs.get(num_query_tokens) if self.config.speculative_method else None
        if sizes and len(seqs) <= sizes[-1]:
            num_padding = next(x for x in sizes if x >= len(seqs)) - len(seqs)
            input_ids += [0] * (num_padding * num_query_tokens)
            positions += [0] * (num_padding * num_query_tokens)
            slot_mapping += [-1] * (num_padding * num_query_tokens)
            context_lens += [0] * num_padding
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs, num_padding)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        context = get_context()
        num_tokens = input_ids.size(0)
        bs = 0 if is_prefill else context.context_lens.size(0)
        # graphs are recorded for 1 token per sequence (plain decode) and 1+k (checking guesses)
        graph_bs = None if is_prefill or self.enforce_eager else self.graph_bs.get(num_tokens // bs)
        if graph_bs is None or bs > graph_bs[-1]:
            return self.model.compute_logits(self.model(input_ids, positions))
        graph = self.graphs[(num_tokens // bs, next(x for x in graph_bs if x >= bs))]
        graph_vars = self.graph_vars
        graph_vars["input_ids"][:num_tokens] = input_ids
        graph_vars["positions"][:num_tokens] = positions
        graph_vars["slot_mapping"].fill_(-1)    # the graph's unused rows must not write KV
        graph_vars["slot_mapping"][:num_tokens] = context.slot_mapping
        graph_vars["context_lens"].zero_()
        graph_vars["context_lens"][:bs] = context.context_lens
        graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
        graph.replay()
        return self.model.compute_logits(graph_vars["outputs"][:num_tokens])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[list[int]]:
        """Returns the new tokens of each sequence (one, or several with speculative decoding)."""
        if not is_prefill and self.config.speculative_method and any(seq.draft_token_ids for seq in seqs):
            return self.run_verify(seqs)
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        if logits is not None and logits.size(0) > len(seqs):    # padding sequences, see prepare_decode
            logits = logits[:len(seqs)]
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return [[token_id] for token_id in token_ids] if self.rank == 0 else None

    @torch.inference_mode()
    def run_verify(self, seqs: list[Sequence]) -> list[list[int]]:
        """Speculative decoding: run the last token plus the guesses through the model in one pass, keep
        the guesses the model agrees with, and add one token chosen by the model."""
        # Always 1+k tokens per sequence, however many guesses there are: fixed shapes let CUDA graphs be
        # reused, and eager runs the very same shapes so both give identical results.
        k = self.config.num_speculative_tokens
        input_ids, positions = self.prepare_decode(seqs, 1 + k)
        drafts = [seq.draft_token_ids + [0] * (k - len(seq.draft_token_ids)) for seq in seqs]
        drafts = torch.tensor(drafts, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True).view(len(seqs), k)
        num_drafts = torch.tensor([len(seq.draft_token_ids) for seq in seqs], dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        temperatures = self.prepare_sample(seqs)
        logits = self.run_model(input_ids, positions, False)[:len(seqs) * (1 + k)]    # drop padding sequences
        num_accepted, next_tokens = verify_drafts(logits, drafts, num_drafts, temperatures)
        reset_context()
        return [seq.draft_token_ids[:a] + [t] for seq, a, t in zip(seqs, num_accepted.tolist(), next_tokens.tolist())]

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        k = config.num_speculative_tokens if config.speculative_method else 0
        max_num_tokens = max(n * sizes[-1] for n, sizes in self.graph_bs.items())
        max_num_seqs = max(sizes[-1] for sizes in self.graph_bs.values())
        max_num_blocks = (config.max_model_len + k + self.block_size - 1) // self.block_size    # checking guesses writes up to k positions past the end
        input_ids = torch.zeros(max_num_tokens, dtype=torch.int64)
        positions = torch.zeros(max_num_tokens, dtype=torch.int64)
        slot_mapping = torch.zeros(max_num_tokens, dtype=torch.int32)
        context_lens = torch.zeros(max_num_seqs, dtype=torch.int32)
        block_tables = torch.zeros(max_num_seqs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_num_tokens, hf_config.hidden_size)
        self.graphs = {}
        self.graph_pool = None

        # biggest first: all graphs share one memory pool, sized by the first capture
        shapes = sorted(((n, bs) for n, sizes in self.graph_bs.items() for bs in sizes), key=lambda x: x[0] * x[1], reverse=True)
        for n, bs in shapes:
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:n * bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:n * bs] = self.model(input_ids[:n * bs], positions[:n * bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:n * bs] = self.model(input_ids[:n * bs], positions[:n * bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[(n, bs)] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
