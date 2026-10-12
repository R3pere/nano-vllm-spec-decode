import numpy as np

from nanovllm.engine.sequence import Sequence


class NgramProposer:
    """Guesses the next tokens by copying from earlier text ("prompt lookup").

    Take the last n tokens, find the most recent earlier place where the same n tokens
    appear, and guess that whatever followed them then will follow them again.
    Tries n = max_n down to min_n and stops at the first match. Runs on the CPU.
    """

    def __init__(self, min_n: int, max_n: int):
        self.min_n = min_n
        self.max_n = max_n
        self.buffers: dict[int, tuple[np.ndarray, int]] = {}    # seq_id -> (token array, number of tokens filled)

    def _tokens(self, seq: Sequence) -> np.ndarray:
        # keep a numpy copy of each sequence and only append the new tokens: converting a
        # few-thousand-token list every step for hundreds of requests costs tens of ms
        buf, filled = self.buffers.get(seq.seq_id, (np.empty(0, dtype=np.int64), 0))
        n = len(seq)
        if n > len(buf):
            new_buf = np.empty(max(2 * len(buf), n + 256), dtype=np.int64)
            new_buf[:filled] = buf[:filled]
            buf = new_buf
        buf[filled:n] = seq.token_ids[filled:n]
        self.buffers[seq.seq_id] = (buf, n)
        return buf[:n]

    def free(self, seq: Sequence):
        self.buffers.pop(seq.seq_id, None)

    def propose(self, seq: Sequence, k: int) -> list[int]:
        if k <= 0:
            return []
        return propose_ngram(self._tokens(seq), k, self.min_n, self.max_n)


def propose_ngram(tokens: np.ndarray, k: int, min_n: int, max_n: int) -> list[int]:
    L = len(tokens)
    # an earlier match of the last n tokens must end at some position e < L-1 with tokens[e] == tokens[-1]
    ends = np.flatnonzero(tokens[:L - 1] == tokens[L - 1])
    for n in range(min(max_n, L - 1), min_n - 1, -1):
        cand = ends[ends >= n - 1]
        for t in range(1, n):
            cand = cand[tokens[cand - t] == tokens[L - 1 - t]]
        if len(cand):
            e = cand[-1]    # most recent match
            return tokens[e + 1: e + 1 + k].tolist()
    return []
