"""Speculative decoding unit tests. CPU only, no model needed.

Covers:
  - n-gram guessing (prompt lookup)
  - the accept/reject rule: greedy cases, plus a statistical test that random sampling with
    guesses produces exactly the model's distribution (and that the test can tell when it doesn't)
  - KV block reservation for several tokens per step
  - scheduler bookkeeping after a step: guesses kept, EOS / max_tokens in the middle of a step,
    crossing block boundaries, and that only blocks with confirmed KV go into the prefix cache

Usage: python tests/test_spec_unit.py
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.ngram_proposer import NgramProposer, propose_ngram
from nanovllm.layers.sampler import verify_drafts


# ---------------------------------------------------------------- n-gram guessing

def test_ngram():
    def p(tokens, k=4, lo=2, hi=4):
        return propose_ngram(np.array(tokens), k, lo, hi)

    assert p([1, 2, 3, 4, 1, 2, 3], k=3) == [4, 1, 2]
    assert p([5, 6, 7, 5, 6, 8, 5, 6]) == [8, 5, 6]               # most recent match wins
    assert p([9, 1, 2, 7, 7, 1, 2, 3, 9, 1, 2]) == [7, 7, 1, 2]    # longer match (9 1 2) wins over a more recent shorter one (1 2)
    assert p([1, 2, 3, 4, 5]) == []                               # no match
    assert p([1, 2, 1, 2], k=10) == [1, 2]                        # limited by the text available after the match
    assert p([1, 2, 3, 1], lo=2) == []                            # 1-token tail is below min_n
    assert p([1, 2, 3, 1], lo=1) == [2, 3, 1]
    assert p([7], lo=1) == []
    assert p([7, 7], lo=1) == [7]

    # the incremental numpy buffer must give the same answers as a fresh array
    rng = np.random.default_rng(0)
    proposer = NgramProposer(2, 4)
    seq = Sequence(rng.integers(0, 5, 20).tolist())
    for _ in range(600):    # grows past several buffer reallocations
        assert proposer.propose(seq, 4) == propose_ngram(np.array(seq.token_ids), 4, 2, 4)
        seq.append_token(int(rng.integers(0, 5)))
    proposer.free(seq)
    assert not proposer.buffers
    print("ngram ok")


# ---------------------------------------------------------------- accept / reject rule

def logits_with_top1(top1, V=10):
    """[S*(K+1), V] logits whose argmax per row is given by top1 [S, K+1]."""
    top1 = torch.tensor(top1)
    logits = torch.randn(*top1.shape, V)
    logits.scatter_(-1, top1.unsqueeze(-1), 100.0)
    return logits.view(-1, V)


def test_verify_greedy():
    drafts = torch.tensor([[3, 4, 5],     # all correct -> keep 3, bonus = top1 after the last guess
                           [3, 9, 5],     # 2nd wrong   -> keep 1, then the model's 4
                           [1, 4, 5],     # 1st wrong   -> keep 0, then the model's 3
                           [3, 4, 5]])    # only 1 real guess: padding must not be accepted even if it matches
    top1 = [[3, 4, 5, 6], [3, 4, 5, 6], [3, 4, 5, 6], [3, 4, 5, 6]]
    num_drafts = torch.tensor([3, 3, 3, 1])
    a, t = verify_drafts(logits_with_top1(top1), drafts, num_drafts, torch.zeros(4))
    assert a.tolist() == [3, 1, 0, 1], a
    assert t.tolist() == [6, 4, 3, 4], t

    # no guesses at all (K = 0) is plain greedy decoding
    a, t = verify_drafts(logits_with_top1([[7], [2]]), torch.zeros(2, 0, dtype=torch.int64), torch.zeros(2, dtype=torch.int64), torch.zeros(2))
    assert a.tolist() == [0, 0] and t.tolist() == [7, 2]

    # greedy and random requests in one batch: greedy ones stay exact
    temps = torch.tensor([0.0, 1.0, 0.0, 1.0])
    for _ in range(20):
        a, t = verify_drafts(logits_with_top1(top1), drafts, num_drafts, temps)
        assert a[0] == 3 and t[0] == 6 and a[2] == 0 and t[2] == 3
    print("verify greedy ok")


def run_trials(p_logits, temperature, draft_probs, n, verify):
    """Run n independent copies of one speculative step with fixed target logits [K+1, V].
    Guesses are sampled from draft_probs [K, V] (one-hot rows = deterministic n-gram guesses)."""
    K = draft_probs.size(0)
    drafts = torch.multinomial(draft_probs, n, replacement=True).T.contiguous()    # [n, K]
    logits = p_logits.unsqueeze(0).expand(n, -1, -1).reshape(n * (K + 1), -1)
    one_hot = bool((draft_probs.max(-1).values == 1).all())
    q = None if one_hot else draft_probs.unsqueeze(0).expand(n, -1, -1)
    a, t = verify(logits, drafts, torch.full((n,), K), torch.full((n,), temperature), q)
    out = torch.cat([drafts, torch.zeros(n, 1, dtype=torch.int64)], 1)
    out[torch.arange(n), a] = t
    return out, a + 1    # output tokens (first a+1 valid) and lengths


def max_tv_distance(p_logits, temperature, out, lengths):
    """For each position j, compare output token j (over trials that got that far) with the model's p_j.
    Speculative sampling promises they are the same distribution. Returns the worst total-variation distance
    and the minimum number of samples it was measured on."""
    V = p_logits.size(-1)
    worst, min_n = 0.0, len(out)
    for j in range(p_logits.size(0)):
        mask = lengths > j
        n = int(mask.sum())
        if n < 20000:
            continue
        emp = torch.bincount(out[mask, j], minlength=V).double() / n
        p = torch.softmax(p_logits[j].double() / temperature, -1)
        worst = max(worst, 0.5 * (emp - p).abs().sum().item())
        min_n = min(min_n, n)
    return worst, min_n


def naive_verify(logits, drafts, num_drafts, temps, q):
    """Deliberately wrong: after a rejection, sample from p instead of p with the guess removed."""
    n, K = drafts.shape
    a, _ = verify_drafts(logits, drafts, num_drafts, temps, q)
    last = logits.view(n, K + 1, -1)[torch.arange(n), a].float() / temps[:, None]
    return a, torch.multinomial(torch.softmax(last, -1), 1).squeeze(1)


def test_verify_distribution():
    torch.manual_seed(0)
    V, K, n, T = 8, 3, 200000, 0.7
    p_logits = torch.randn(K + 1, V) * 1.5
    tol = 0.02    # sampling noise for 8 outcomes and >= 20k samples is under 0.01

    # n-gram style: fixed guesses. Pick guesses the model likes moderately so both accepts and rejects happen.
    p = torch.softmax(p_logits / T, -1)
    guesses = (p - 0.3).abs().argmin(-1)[:K]
    one_hot = torch.nn.functional.one_hot(guesses, V).float()
    # draft-model style: guesses sampled from some other distribution q
    q = torch.softmax(p_logits[:K] / T + torch.randn(K, V), -1)

    for name, draft_probs in (("fixed guesses", one_hot), ("sampled guesses", q)):
        out, lengths = run_trials(p_logits, T, draft_probs, n, verify_drafts)
        tv, m = max_tv_distance(p_logits, T, out, lengths)
        print(f"  {name}: worst TV distance {tv:.4f} over >= {m} samples, mean tokens per step {lengths.float().mean():.2f}")
        assert tv < tol, (name, tv)

        out, lengths = run_trials(p_logits, T, draft_probs, n, naive_verify)
        tv_bad, _ = max_tv_distance(p_logits, T, out, lengths)
        print(f"  {name}, deliberately wrong rule: worst TV distance {tv_bad:.4f}")
        assert tv_bad > 3 * tol, (name, tv_bad)    # the test must be able to catch a wrong rule
    print("verify distribution ok")


# ---------------------------------------------------------------- blocks and scheduler

BS = 4    # tiny blocks so crossing block boundaries is easy


def make_scheduler(num_blocks=32, k=3, eos=99, max_model_len=1000):
    Sequence.block_size = BS
    config = SimpleNamespace(max_num_seqs=8, max_num_batched_tokens=64, eos=eos, max_model_len=max_model_len, kvcache_block_size=BS,
                             num_kvcache_blocks=num_blocks, speculative_method="ngram" if k else None,
                             num_speculative_tokens=k)
    return Scheduler(config)


def check_blocks(sched: Scheduler):
    bm = sched.block_manager
    refs = {}
    for seq in sched.running:
        for b in seq.block_table:
            refs[b] = refs.get(b, 0) + 1
    for b, r in refs.items():
        assert bm.blocks[b].ref_count == r, (b, bm.blocks[b].ref_count, r)
    assert set(refs) == bm.used_block_ids
    assert set(bm.free_block_ids).isdisjoint(bm.used_block_ids)
    assert len(bm.free_block_ids) + len(bm.used_block_ids) == len(bm.blocks)
    assert all(bm.blocks[b].ref_count == 0 for b in bm.free_block_ids)
    for seq in sched.running:
        # the invariant between steps: everything but the last token has its KV cached
        assert seq.num_cached_tokens == seq.num_tokens - 1, (seq.num_cached_tokens, seq.num_tokens)
        # only blocks fully covered by confirmed KV may be in the prefix cache, with matching tokens
        for i, b in enumerate(seq.block_table):
            block = bm.blocks[b]
            if block.hash != -1:
                assert (i + 1) * BS <= seq.num_cached_tokens, (i, seq.num_cached_tokens)
                assert block.token_ids == seq.block(i)
                assert bm.hash_to_block_id[block.hash] == b


def test_block_reservation():
    bm = BlockManager(16, BS)
    Sequence.block_size = BS
    # with one token per step this must behave exactly like upstream: a new block when len % BS == 1
    for n in range(1, 30):
        seq = Sequence(list(range(n)))
        seq.block_table = list(range((n - 1 + BS - 1) // BS))    # what upstream holds between steps
        assert bm.num_blocks_to_append(seq, 1) == (n % BS == 1)
    # 1+k tokens: the table must cover positions up to len-2+num_tokens
    seq = Sequence(list(range(7)))    # positions 6..9 are written -> 3 blocks
    seq.block_table = [bm._allocate_block(), bm._allocate_block()]
    assert bm.num_blocks_to_append(seq, 4) == 1
    bm.may_append(seq, 4)
    assert len(seq.block_table) == 3 and bm.num_blocks_to_append(seq, 4) == 0
    seq.block_table.append(bm._allocate_block())    # already over-reserved: nothing more needed
    assert bm.num_blocks_to_append(seq, 4) == 0 and bm.can_append(seq, 4)
    bm.deallocate(seq)
    assert len(bm.free_block_ids) == 16 and not bm.used_block_ids
    print("block reservation ok")


def prefill(sched, prompt, max_tokens=50, ignore_eos=False, first_token=50):
    seq = Sequence(prompt, SamplingParams(temperature=0, max_tokens=max_tokens, ignore_eos=ignore_eos))
    sched.add(seq)
    seqs, is_prefill = sched.schedule()
    assert is_prefill and seqs == [seq]
    sched.postprocess(seqs, [[first_token]], True)
    check_blocks(sched)
    return seq


def decode(sched, seq, drafts, new_tokens):
    seqs, is_prefill = sched.schedule()
    assert not is_prefill and seq in seqs
    # the scheduler must have reserved KV room for the last token plus k guesses
    assert len(seq.block_table) * BS >= len(seq) - 1 + 1 + sched.num_spec_tokens
    seq.draft_token_ids = drafts
    n = sched.postprocess([seq], [new_tokens], False)
    check_blocks(sched)
    return n


def test_scheduler():
    sched = make_scheduler()
    seq = prefill(sched, [1, 2, 3, 4, 5, 6])    # 6 prompt tokens + 1 sampled = 7 tokens, 6 cached
    assert seq.num_tokens == 7

    # 2 of 3 guesses kept + 1 model token = 3 new tokens; crosses into the 3rd block
    assert decode(sched, seq, [7, 8, 9], [7, 8, 60]) == 3
    assert seq.token_ids[-4:] == [50, 7, 8, 60] and seq.num_cached_tokens == 9
    # nothing kept: behaves like plain decode
    assert decode(sched, seq, [1, 1, 1], [61]) == 1 and seq.num_cached_tokens == 10
    # all kept + bonus token: 4 tokens in one step, crossing another block boundary
    assert decode(sched, seq, [62, 63, 64], [62, 63, 64, 65]) == 4 and seq.num_cached_tokens == 14
    assert seq.token_ids[6:] == [50, 7, 8, 60, 61, 62, 63, 64, 65]

    # EOS in the middle of the step: the rest is dropped and the request finishes
    assert decode(sched, seq, [66, 99, 67], [66, 99, 67, 68]) == 2
    assert seq.is_finished and seq.completion_token_ids[-2:] == [66, 99]
    assert not sched.running and not seq.block_table
    assert len(sched.block_manager.free_block_ids) == 32

    # max_tokens reached in the middle of the step
    seq = prefill(sched, [1, 2, 3, 4, 5, 6], max_tokens=3)
    assert decode(sched, seq, [70, 71, 72], [70, 71, 72, 73]) == 2
    assert seq.is_finished and seq.completion_token_ids == [50, 70, 71]

    # ignore_eos keeps going past EOS
    seq = prefill(sched, [1, 2, 3, 4, 5, 6], ignore_eos=True)
    assert decode(sched, seq, [99, 99], [99, 99, 99]) == 3 and not seq.is_finished
    sched.running.remove(seq)
    sched.block_manager.deallocate(seq)

    # a request never grows past max_model_len, even when more guesses were kept
    sched = make_scheduler(max_model_len=10)
    seq = prefill(sched, [1, 2, 3, 4, 5, 6])    # 7 tokens
    assert decode(sched, seq, [70, 71, 72], [70, 71, 72, 73]) == 3
    assert seq.is_finished and len(seq) == 10
    print("scheduler ok")


def test_prefix_cache_after_spec():
    """Blocks finished by speculative steps are reused by a later request with the same text; the
    reused blocks must hold exactly the tokens that were finally kept."""
    sched = make_scheduler()
    seq = prefill(sched, [1, 2, 3, 4, 5, 6])
    decode(sched, seq, [7, 8, 9], [7, 8, 60])
    decode(sched, seq, [10, 11, 12], [10, 11, 12, 13])
    decode(sched, seq, [99], [99])    # finish
    assert seq.is_finished
    text = seq.token_ids    # 1..6, 50, 7, 8, 60, 10, 11, 12, 13, 99 (15 tokens)
    bm = sched.block_manager
    # KV was confirmed for the first 14 tokens -> blocks 0..2 (tokens 0..11) are hashed, block 3 is not
    seq2 = Sequence(text + [5, 5])
    assert bm.can_allocate(seq2) == 3
    seq3 = Sequence(text[:9] + [77] * 8)    # diverges inside block 2 -> only blocks 0, 1 reusable
    assert bm.can_allocate(seq3) == 2
    print("prefix cache ok")


def test_preemption_with_reservation():
    # 2 requests, 5 blocks of 4 tokens: reserving 1+k slots must preempt instead of running out of blocks
    sched = make_scheduler(num_blocks=5, k=3)
    a = prefill(sched, [1, 2, 3, 4, 5, 6])
    b = prefill(sched, [11, 12, 13, 14, 15, 16])
    preempted = False
    for _ in range(8):
        seqs, is_prefill = sched.schedule()
        preempted |= any(s.status == SequenceStatus.WAITING and not s.block_table for s in (a, b))
        sched.postprocess(seqs, [[seq.last_token + 1] for seq in seqs], is_prefill)
        check_blocks(sched)
    assert preempted
    print("preemption ok")


if __name__ == "__main__":
    test_ngram()
    test_verify_greedy()
    test_verify_distribution()
    test_block_reservation()
    test_scheduler()
    test_prefix_cache_after_spec()
    test_preemption_with_reservation()
    print("PASS")
