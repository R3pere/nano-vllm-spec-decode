from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        # speculative decoding: each decode step checks up to k guessed tokens, so reserve KV room for 1+k tokens
        self.num_spec_tokens = config.num_speculative_tokens if config.speculative_method else 0
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        num_tokens = 1 + self.num_spec_tokens
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            if self.num_spec_tokens and num_batched_tokens + num_tokens > self.max_num_batched_tokens:
                break
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq, num_tokens):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = num_tokens
                seq.is_prefill = False
                self.block_manager.may_append(seq, num_tokens)
                num_batched_tokens += num_tokens
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[list[int]], is_prefill: bool) -> int:
        """token_ids[i] are the new tokens of seqs[i]: one token normally, or with speculative decoding
        the accepted guesses followed by one token from the model. Returns how many tokens were appended."""
        num_appended_total = 0
        for seq, new_token_ids in zip(seqs, token_ids):
            seq.draft_token_ids = []
            if is_prefill:
                self.block_manager.hash_blocks(seq)
                seq.num_cached_tokens += seq.num_scheduled_tokens
                seq.num_scheduled_tokens = 0
                if seq.num_cached_tokens < seq.num_tokens:
                    continue
            num_appended = 0
            for token_id in new_token_ids:
                seq.append_token(token_id)
                num_appended += 1
                # one step can produce several tokens, so EOS or max_tokens may be hit midway: drop the rest
                if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                    seq.status = SequenceStatus.FINISHED
                    break
            num_appended_total += num_appended
            if not is_prefill:
                # Valid KV this step: the old last token and the guesses that were kept (all appended tokens
                # except the newest one, whose KV is computed next step). Moving num_cached_tokens forward by
                # exactly that much is the whole "rollback": KV written for rejected guesses lies past it and
                # gets overwritten next step. Hash only after appending, so blocks are hashed from final tokens.
                seq.num_scheduled_tokens = num_appended
                self.block_manager.hash_blocks(seq)
                seq.num_cached_tokens += num_appended
                seq.num_scheduled_tokens = 0
            if seq.is_finished:
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
        return num_appended_total
