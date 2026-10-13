"""Choosing k (how many guesses to check) every step, from how often guesses get accepted.

Checking k guesses makes a step more expensive (the model reads 1+k tokens per request instead of 1), and it
only pays off when enough guesses are accepted. How much more expensive depends on the batch size and the GPU,
so it is measured once at startup (ModelRunner.measure_step_costs): costs[k][i] = milliseconds of one step
with k guesses at the i-th batch-size bucket.

Each request keeps a running estimate alpha of "probability that a guess is accepted". If guesses are accepted
independently with probability alpha, a step with m guesses produces on average
    1 + alpha + alpha^2 + ... + alpha^m = (1 - alpha^(m+1)) / (1 - alpha)
tokens (the guesses kept until the first rejection, plus the one token the model always adds).
For the whole batch we pick the k with the most tokens per millisecond:
    sum over requests of expected_tokens(alpha, min(k, guesses available)) / costs[k][bucket]
k = 0 means plain decoding (1 token per request).
"""

PRIOR_ACCEPTED = 1.0    # a new request starts with these counts: alpha = 1 / (1 + 1) = 0.5, so it gets tried.
PRIOR_REJECTED = 1.0    # They fade like any observation, so a request whose guesses are always kept reaches ~1.
DECAY = 0.8             # after each checked step, older observations count less (looks back ~5 steps)
IDLE_DECAY = 0.95       # after a step without checking, the counts move slowly back to the prior, so a request
                        # that stopped guessing well gets tried again later
PROBE_INTERVAL = 16     # draft model: after this many steps in a row with k = 0, check the smallest k once anyway.
                        # In big batches k = 0 can win at the starting alpha 0.5, and without a check alpha is never
                        # measured, so it would stay 0.5 (and k = 0) for good. One cheap step in 16 costs little.


def expected_tokens(alpha: float, m: int) -> float:
    """Average tokens one step produces for a request with m guesses, each accepted with probability alpha."""
    if alpha >= 1.0:
        return m + 1.0
    return (1.0 - alpha ** (m + 1)) / (1.0 - alpha)


def acceptance(accepted: float, rejected: float) -> float:
    """Estimate of alpha from (decayed) counts of accepted guesses and of rejections."""
    total = accepted + rejected
    return accepted / total if total > 0 else PRIOR_ACCEPTED / (PRIOR_ACCEPTED + PRIOR_REJECTED)


def update_counts(accepted: float, rejected: float, num_drafts: int, num_accepted: int) -> tuple[float, float]:
    """New (accepted, rejected) counts after one step. A step with m guesses where a were accepted is a run of
    Bernoulli trials: a successes, then one failure if a < m (the guesses after the first rejection were never
    tested). This counting gives the maximum-likelihood alpha for the model above."""
    if num_drafts == 0:    # nothing was checked this step
        return (IDLE_DECAY * accepted + (1 - IDLE_DECAY) * PRIOR_ACCEPTED,
                IDLE_DECAY * rejected + (1 - IDLE_DECAY) * PRIOR_REJECTED)
    return accepted * DECAY + num_accepted, rejected * DECAY + (num_accepted < num_drafts)


def choose_k(alphas: list[float], max_drafts: list[int], costs: dict[int, list[float]], buckets: dict[int, list[int]]) -> int:
    """Pick k for one decode step.

    alphas[i]     acceptance estimate of request i
    max_drafts[i] most guesses request i can use this step (fewer near max_tokens; with n-gram, the length of
                  the guess it actually found, possibly 0)
    costs[k]      measured step time (any unit) for each batch-size bucket of k; k = 0 must be present
    buckets[k]    the batch-size buckets costs[k] refers to (ascending); k is only allowed if the batch fits
    """
    bs = len(alphas)
    best_k, best_rate = 0, -1.0
    for k in sorted(costs):
        sizes = buckets[k]
        if bs > sizes[-1]:
            continue
        cost = costs[k][next(i for i, x in enumerate(sizes) if x >= bs)]
        tokens = sum(expected_tokens(a, min(k, m)) for a, m in zip(alphas, max_drafts))
        if tokens / cost > best_rate:
            best_k, best_rate = k, tokens / cost
    return best_k


def probe(k: int, ks: list[int], idle_steps: int) -> tuple[int, int]:
    """Draft model: the k to use given choose_k's pick k, and the new count of steps in a row that chose k = 0
    although guesses were possible. Every PROBE_INTERVAL-th such step uses the smallest allowed k > 0 instead,
    so acceptance keeps being measured (see PROBE_INTERVAL)."""
    if k > 0:
        return k, 0
    idle_steps += 1
    if idle_steps >= PROBE_INTERVAL:
        return min(x for x in ks if x > 0), 0
    return 0, idle_steps
